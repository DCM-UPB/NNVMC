#include <iostream>
#include <cmath>
#include <stdexcept>

#include "vmc/WaveFunction.hpp"
#include "vmc/Hamiltonian.hpp"
#include "vmc/VMC.hpp"
#include "vmc/StochasticReconfigurationOptimization.hpp"
#include "ffnn/FeedForwardNeuralNetwork.hpp"
#include "ffnn/ActivationFunctionManager.hpp"
#include "ffnn/PrintUtilities.hpp"
#include "ffnn/SmartBetaGenerator.hpp"
#include "FFNNWaveFunction.hpp"

/*
  Hamiltonian describing a n-particle harmonic oscillator:
  H  =  Sum_i^n( p_i^2 ) /2m  +  1/2 * w^2 * Sum_i^n( x_i^2 )
*/
class HarmonicTrapSoftCore1DNP: public Hamiltonian{

protected:
    const int _npart;
    const double _wsqh;
    const double _a, _b;

public:
    HarmonicTrapSoftCore1DNP(const double w, const double a, const double b, FFNNWaveFunction * wf):
        Hamiltonian(1 /*num space dimensions*/, wf->getNPart() /*num particles*/, wf),
        _npart(wf->getNPart()), _wsqh(0.5 * w *w), _a(a), _b(b) {}

    // potential energy
    double localPotentialEnergy(const double *r)
    {
        double pot = 0.;
        for(int i = 0; i<_npart; ++i) {
            pot += _wsqh*r[i]*r[i];
            for (int j=i+1; j<_npart; ++j) {
                double dist=abs(r[i]-r[j]);
                if (dist < _a) pot += _b;
            }
        }
        return pot;
    }
};

int main(){
    using namespace std;

    const int NSPACEDIM = 1;
    const int NPARTICLES = 2;
    /*
    const int NHIDDENLAYERS = 2;
    const int HIDDENLAYERSIZE[NHIDDENLAYERS] = {6,3};
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(NSPACEDIM*NPARTICLES + 1, HIDDENLAYERSIZE[0], 2);
    ffnn->storeOnFile("ffnn_1.txt");
    for (int i=1; i<NHIDDENLAYERS; ++i){
        ffnn->pushHiddenLayer(HIDDENLAYERSIZE[i]);
    }
    ffnn->storeOnFile("ffnn_2.txt");
    ffnn->connectFFNN();
    ffnn->storeOnFile("ffnn_3.txt");
    ffnn->assignVariationalParameters();
    ffnn->storeOneFile("ffnn_4.txt");

    //Set ACTFs for hidden units
    for (int i=0; i<NHIDDENLAYERS; ++i) {
        for (int j=0; j<HIDDENLAYERSIZE[i]-1; ++j) {
            ffnn->getNNLayer(i)->getNNUnit(j)->setActivationFunction(std_actf::provideActivationFunction("TANS"));
        }
        //ffnn->getNNLayer(i)->getOffsetUnit()->setProtoValue(0.);
    }

    //Set ACTF for output unit
    ffnn->getNNLayer(NHIDDENLAYERS)->getNNUnit(0)->setActivationFunction(std_actf::provideActivationFunction("LGS"));

    smart_beta::generateSmartBeta(ffnn);

    ffnn->getLayer(ffnn->getNLayers()-2)->getOffsetUnit()->setProtoValue(0.); // disable output offset
    //ffnn->getOutputLayer()->getOutputNNUnit(0)->setScale(1.05); // allow the lgs a bit of freedom
    ffnn->storeOnFile("ffnn_5.txt");

    cout << "Created FFNN with " << NHIDDENLAYERS << " hidden layer(s) of " << HIDDENLAYERSIZE[0] << ", " << HIDDENLAYERSIZE[1] << " units each." << endl << endl;
    */
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork("nn.in");
    ffnn->getLayer(ffnn->getNLayers()-2)->getOffsetUnit()->setProtoValue(0.); // disable output offset

    // Declare the trial wave functions
    FFNNWaveFunction * psi = new FFNNWaveFunction(NSPACEDIM, NPARTICLES, ffnn, true, false, false);

    // Store in two files the the initial wf, one for plotting, and for recovering it as nn
    cout << "Writing the plot file of the initial wave function in (plot_)init_wf.txt" << endl << endl;
    double * base_input = new double[psi->getBareFFNN()->getNInput()];
    for (int i=0; i<psi->getBareFFNN()->getNInput(); ++i) base_input[i] = 0.;
    const double min = -5.;
    const double max = 5.;
    const int npoints = 100;
    writePlotFile(psi->getBareFFNN(), base_input, 0, 0, min, max, npoints, "getOutput", "plot_init_wf_r1.txt");
    writePlotFile(psi->getBareFFNN(), base_input, 1, 0, min, max, npoints, "getOutput", "plot_init_wf_r2.txt");
    psi->getBareFFNN()->storeOnFile("wf_init.txt");


    // Declare an Hamiltonian
    // We use the harmonic oscillator with w=1 and w=2
    double w1 = 1.0;
    double w2 = 0.5;

    HarmonicTrapSoftCore1DNP * ham = new HarmonicTrapSoftCore1DNP(w1, 0.5, 20., psi);

    cout << endl << " - - - FFNN-WF FUNCTION OPTIMIZATION - - - " << endl << endl;

    VMC * vmc; // VMC object we will resuse
    const long E_NMC = 100000l; // MC samplings to use for computing the energy
    cout << "E_NMC = " << E_NMC << endl << endl;
    double energy[4]; // energy
    double d_energy[4]; // energy error bar

    cout << "-> ham1:    w = " << w1 << endl << endl;
    vmc = new VMC(psi, ham);

    // set an integration range, because the NN might be completely delocalized
    double ** irange = new double*[NSPACEDIM*NPARTICLES];
    for (int i=0; i<NSPACEDIM*NPARTICLES; ++i) {irange[i] = new double[2]; irange[i][0] = -10.; irange[i][1] = 10.;}
    cout << "Integration range: " << irange[0][0] << "   <->   " << irange[0][1] << endl << endl;
    vmc->getMCI()->setIRange(irange);


    cout << "   Starting energy:" << endl;
    vmc->computeVariationalEnergy(E_NMC, energy, d_energy);
    cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
    cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
    cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
    cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl;

    cout << "   Optimization . . ." << endl;
    //vmc->nmsimplexOptimization(E_NMC, 1.0, 0.1, 0.005);
    vmc->stochasticReconfigurationOptimization(E_NMC);
    cout << "   . . . Done!" << endl << endl;

    cout << "   Optimized energy:" << endl;
    vmc->computeVariationalEnergy(E_NMC, energy, d_energy);
    cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
    cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
    cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
    cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl << endl;


    // Store in two files the the initial wf, one for plotting, and for recovering it as nn

    cout << "Writing the plot file of the optimised wave function in (plot_)opt_wf.txt" << endl << endl;
    writePlotFile(psi->getBareFFNN(), base_input, 0, 0, min, max, npoints, "getOutput", "plot_opt_wf_r1.txt");
    writePlotFile(psi->getBareFFNN(), base_input, 1, 0, min, max, npoints, "getOutput", "plot_opt_wf_r2.txt");
    psi->getBareFFNN()->storeOnFile("wf_opt.txt");


    for (int i=0; i<NSPACEDIM*NPARTICLES; ++i) delete [] irange[i];
    delete[] irange;
    delete vmc;
    delete ham;
    delete base_input;
    delete psi;
    delete ffnn;

    return 0;
}
