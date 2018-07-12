#include <iostream>
#include <cmath>
#include <stdexcept>

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"
#include "ReadUtilities.hpp"
#include "WaveFunction.hpp"
#include "FFNNWaveFunction.hpp"
#include "Hamiltonian.hpp"
#include "VMC.hpp"



/*
  Hamiltonian describing a 1-particle harmonic oscillator:
  H  =  p^2 / 2m  +  1/2 * w^2 * x^2
*/
class HarmonicOscillator1D1P: public Hamiltonian{

protected:
    double _w;

public:
    HarmonicOscillator1D1P(const double w, WaveFunction * wf):
        Hamiltonian(1 /*num space dimensions*/, 1 /*num particles*/, wf) {_w=w;}

    // potential energy
    double localPotentialEnergy(const double *r)
    {
        return (0.5*_w*_w*(*r)*(*r));
    }
};


/*
  Gaussian Trial Wave Function for 1 particle in 1 dimension, that uses two variational parameters: x0 and v (variance).
  Psi  =  exp( -0.5*(x-x0)^2/v )
  Notice that the corresponding probability density (sampling function) is Psi^2.
  The parameter eps controls the minimal possible value for v
*/
class Gaussian1D1POrbital: public WaveFunction{
protected:
    double _x0, _v, _niv, _eps;

public:
    Gaussian1D1POrbital(const double &x0, const double &v, const double &eps):
        WaveFunction(1 /*num space dimensions*/, 1 /*num particles*/, 1 /*num wf components*/, 2 /*num variational parameters*/, false, false, false) {_eps=eps; this->setVP(x0, v);}

    // overwrite the parent setVP
    void setVP(const double *in){
        setVP(in[0], in[1]);
    }

    //add another own setVP for convenience
    void setVP(const double &x0, const double &v) {
        _x0=x0;
        _v=(v<_eps)? _eps:v;
        _niv = -1./_v;
    }

    void getVP(double *out){
        out[0]=_x0; out[1]=_v;
    }

    void samplingFunction(const double *x, double *out){
        /*
          Compute the sampling function proto value, used in getAcceptance()
        */
        *out = _niv*(x[0]-_x0)*(x[0]-_x0);
    }

    double getAcceptance(const double * protoold, const double * protonew){
        /*
          Compute the acceptance probability
        */
        return exp(protonew[0]-protoold[0]);
    }

    void computeAllDerivatives(const double *x){
        _setD1DivByWF(0, _niv*(x[0]-_x0));
        _setD2DivByWF(0, _niv + _niv*_niv*(x[0]-_x0)*(x[0]-_x0));
        if (hasVD1()){
            _setVD1DivByWF(0, -_niv*(x[0]-_x0));
            _setVD1DivByWF(1, 0.5*_niv*_niv*(x[0]-_x0)*(x[0]-_x0));
        }
    }

};


int main(){
    using namespace std;

    // Declare some trial wave functions
    Gaussian1D1POrbital * psig = new Gaussian1D1POrbital(1.0, 1.0, 0.1);

    //   vector<vector<string>> actf;
    //readFFNNStructure("nn.txt", actf);

    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork("nn.in");

    printFFNNStructure(ffnn);

    // Declare the trial wave functions
    FFNNWaveFunction * psin = new FFNNWaveFunction(1, 1, ffnn, false, false, false);

    // Declare an Hamiltonian
    // We use the harmonic oscillator with w=1 and w=2
    const double w = 1.;
    HarmonicOscillator1D1P * hamg = new HarmonicOscillator1D1P(w, psig);
    HarmonicOscillator1D1P * hamn = new HarmonicOscillator1D1P(w, psin);

    VMC * vmcg = new VMC(psig, hamg);
    VMC * vmcn = new VMC(psin, hamn);

    double ** irange = new double*[1];
    irange[0] = new double[2];
    irange[0][0] = -5.;
    irange[0][1] = 5.;
    cout << "Integration range: " << irange[0][0] << "   <->   " << irange[0][1] << endl << endl;
    vmcg->getMCI()->setIRange(irange);
    vmcn->getMCI()->setIRange(irange);


    const long NMC = 200000l; // MC samplings to use for computing the energy
    double * energy = new double[4]; // energy
    double * d_energy = new double[4]; // energy error bar
    double * vpg = new double[psig->getNVP()];
    double * vpn = new double[psin->getNVP()];


    cout << "-> ham:    w = " << w << endl << endl;

    psig->getVP(vpg);
    psin->getVP(vpn);
    cout << "Initial Wave Function parameters:" << endl;
    cout << endl;
    cout << "Gaussian:" << endl;
    cout << "    x0 = " << vpg[0] << endl;
    cout << "    v  = " << vpg[1] << endl;
    cout << endl;
    cout << "Neural Network:" << endl;
    for (int i=0; i<psin->getNVP(); ++i) cout << "       b" << i << " = " << vpn[i] << endl;
    cout << endl;

    vmcg->computeVariationalEnergy(NMC, energy, d_energy);
    cout << "Gaussian energies:" << endl;
    cout << "    Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
    cout << "    Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
    cout << "    Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
    cout << "    Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl;

    vmcn->computeVariationalEnergy(NMC, energy, d_energy);
    cout << "Neural Network energies:" << endl;
    cout << "    Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
    cout << "    Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
    cout << "    Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
    cout << "    Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl;


    delete[] vpg;
    delete[] vpn;
    delete[] d_energy;
    delete[] energy;
    delete vmcg;
    delete vmcn;
    delete hamg;
    delete hamn;
    delete psig;
    delete psin;
    delete ffnn;


    return 0;
}
