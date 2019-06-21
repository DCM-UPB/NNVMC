#include <iostream>
#include <cmath>

#include "SannifaWaveFunction.hpp"
#include "FFNNWaveFunction.hpp"
#include "vmc/VMC.hpp"
#include "vmc/Hamiltonian.hpp"
//#include "vmc/MPIVMC.hpp"
//#include "mci/MPIMCI.hpp"
#include "nfm/LogNFM.hpp"
#include "ffnn/FeedForwardNeuralNetwork.hpp"
#include "sannifa/ANNFunctionInterface.hpp"
#include "sannifa/TorchNetwork.hpp"
#include "sannifa/FFNNetwork.hpp"

#include <torch/torch.h>

/*
Hamiltonian describing a n-particle harmonic oscillator:
   H  =  Sum_i^n( p_i^2 ) /2m  +  1/2 * w^2 * Sum_i^n( x_i^2 )
*/
class HarmonicOscillatorNDNP: public Hamiltonian{

protected:
   int _n;
   double _wsqh;

public:
   HarmonicOscillatorNDNP(const double w, WaveFunction * wf, const bool usePBKE = true):
     Hamiltonian(wf->getNSpaceDim() /*num space dimensions*/, wf->getNPart() /*num particles*/, wf, usePBKE) {
     _n = wf->getNSpaceDim() * wf->getNPart();
     _wsqh = 0.5 * w * w;
   }

   // potential energy
   double localPotentialEnergy(const double *r)
   {
      double pot = 0.;
      for(int i = 0; i<_n; ++i) {
          pot += _wsqh*r[i]*r[i];
      }
      return pot;
   }
};


struct Model: public torch::nn::Module {

    // Constructor
    Model(const int insize, const int hsize, const int outsize)
    {
        // construct and register your layers
        in = register_module("in",torch::nn::Linear(insize,hsize));
        h = register_module("h",torch::nn::Linear(hsize,hsize));
        out = register_module("out",torch::nn::Linear(hsize,outsize));
        this->to(torch::kFloat64);
    }

    Model(const Model &other)
    {
        in = register_module("in", other.in);
        h = register_module("h", other.h);
        out = register_module("out", other.out);
        //this->to(torch::kFloat64);
    }

    // the forward operation (how data will flow from layer to layer)
    torch::Tensor forward(torch::Tensor X)
    {
        // let's pass relu 
        X = torch::celu(in->forward(X));
        X = torch::celu(h->forward(X));
        X = torch::softplus(out->forward(X));
        
        // return the output
        return X;
    }

    const std::shared_ptr<Model> clone(c10::optional<c10::Device>&)
    {
        std::shared_ptr<Model> ptr (new Model((*this)));
        return ptr;
    }

    torch::nn::Linear in{nullptr},h{nullptr},out{nullptr};

};


int main(int argc, char **argv){
    int myrank = MPIVMC::Init(); // first run custom MPI init

    using namespace std;

    // options
    const long E_NMC = 100000l; // for energy evaluation at start/end
    const long G_NMC = 25000l; // for energy/gradient evaluations during optimization
    const int max_n_const = 100; // constant values to stop
    const int nparticles = 16; // number of (independent) particles

    if (myrank == 0) {
        cout << "E_NMC = " << E_NMC << endl;
        cout << "G_NMC = " << G_NMC << endl;
        cout << "maxn const = " << max_n_const << endl;
        cout << "nparticles = " << nparticles << endl;
    }


    // create pyTorch ANN
    torch::nn::AnyModule anymodel(Model(nparticles, nparticles*6, 1));
    TorchNetwork wrappedANN(anymodel, nparticles, 1);

/*
    // create libffnn ANN
    FeedForwardNeuralNetwork ffnn(nparticles+1, nparticles*6+1, 2);
    ffnn.pushHiddenLayer(nparticles*6+1);
    ffnn.getOutputLayer()->getNNUnit(0)->setActivationFunction("LGS");
    ffnn.connectFFNN();
    ffnn.assignVariationalParameters();
    FFNNetwork wrappedANN(&ffnn);
*/

    wrappedANN.enableFirstDerivative();
    wrappedANN.enableSecondDerivative();
    wrappedANN.enableVariationalFirstDerivative();

    // Declare the trial wave function
    SannifaWaveFunction * psi = new SannifaWaveFunction(1 /*space dimension*/, nparticles, &wrappedANN);
    
    // do the same with old wrapper
    //FFNNWaveFunction * psi = new FFNNWaveFunction(1, nparticles, &ffnn, true);
    if (myrank == 0) {
        cout << "wrappedANN.getNInput() -> " << wrappedANN.getNInput() << endl;
        cout << "wrappedANN.getNOutput() -> " << wrappedANN.getNOutput() << endl;
        cout << "wrappedANN.getNVariationalParameters() -> " << wrappedANN.getNVariationalParameters() << endl;

        cout << "psi->getNSpaceDim() -> " << psi->getNSpaceDim() << endl;
        cout << "psi->getNPart() -> " << psi->getNPart() << endl;
    }

    // Declare an Hamiltonian
    // We use the harmonic oscillator with w=1 and soft-core potential
    HarmonicOscillatorNDNP * ham = new HarmonicOscillatorNDNP(1., psi, wrappedANN.hasSecondDerivative());


    NFMLogManager * log_manager;
    if (myrank == 0) {
        log_manager = new NFMLogManager();
        log_manager->setLoggingOn();
    }

    if (myrank == 0) cout << endl << " - - - ANN-WF FUNCTION OPTIMIZATION - - - " << endl << endl;

    VMC * vmc = new VMC(psi, ham); // VMC object we will resuse
    double energy[4]; // energy
    double d_energy[4]; // energy error bar


    // set an integration range, because the NN might be completely delocalized
    const double hrange = 5.;
    double ** irange = new double*[nparticles];
    for (int i=0; i<nparticles; ++i) {
        irange[i] = new double[2];
        irange[i][0] = -hrange;
        irange[i][1] = hrange;
    }
    if (myrank == 0) cout << "Integration range: " << irange[0][0] << "   <->   " << irange[0][1] << endl << endl;
    vmc->getMCI()->setIRange(irange);

    const double target_acc_rate = 0.5;
    vmc->getMCI()->setTargetAcceptanceRate(&target_acc_rate);

    if (myrank == 0) cout << "   Starting energy:" << endl;
    vmc->computeVariationalEnergy(E_NMC, energy, d_energy);
    if (myrank == 0) {
        cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
        cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
        cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
        cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl;

        cout << "   Optimization . . ." << endl;
    }

    vmc->adamOptimization(G_NMC, false, false, max_n_const, false, 0.005, 0.01, 0.9, 0.9);
    if (myrank == 0) {
        cout << "   . . . Done!" << endl << endl;

        cout << "   Optimized energy:" << endl;
    }

    vmc->computeVariationalEnergy(E_NMC, energy, d_energy);

    if (myrank == 0) {
        cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
        cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
        cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
        cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl << endl;
    }

    for (int i=0; i<nparticles; ++i) delete [] irange[i];
    delete[] irange;

    delete vmc;
    delete ham;
    delete psi;
    if (myrank == 0) delete log_manager;

    MPIVMC::Finalize();

    return 0;
}

