#include <iostream>
#include <cmath>
#include <stdexcept>

#include "WaveFunction.hpp"
#include "Hamiltonian.hpp"
#include "VMC.hpp"
#include "ConjGrad.hpp"
#include "FFNNWaveFunction.hpp"

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"



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





int main(){
    using namespace std;

    const int HIDDENLAYERSIZE = 15;
    const int NHIDDENLAYERS = 1;
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(2, HIDDENLAYERSIZE, 2);
    for (int i=0; i<NHIDDENLAYERS-1; ++i){
        ffnn->pushHiddenLayer(HIDDENLAYERSIZE);
    }
    ffnn->connectFFNN();
    ffnn->assignVariationalParameters();

    // Declare the trial wave functions
    FFNNWaveFunction * psi = new FFNNWaveFunction(1, 1, ffnn, true, false, false);

    // Store in a .txt file the values of the initial wf, so that it is possible to plot it
    cout << "Writing the plot file of the initial wave function in plot_init_wf.txt" << endl << endl;
    double * base_input = new double[psi->getBareFFNN()->getNInput()]; // no need to set it, since it is 1-dim
    const int input_i = 0;
    const int output_i = 0;
    const double min = -7.5;
    const double max = 7.5;
    const int npoints = 500;
    writePlotFile(psi->getBareFFNN(), base_input, input_i, output_i, min, max, npoints, "getOutput", "plot_init_wf.txt");


    // Declare an Hamiltonian
    // We use the harmonic oscillator with w=1 and w=2
    const double w1 = 1.;
    HarmonicOscillator1D1P * ham = new HarmonicOscillator1D1P(w1, psi);




    cout << endl << " - - - FFNN-WF FUNCTION OPTIMIZATION - - - " << endl << endl;

    VMC * vmc; // VMC object we will resuse
    const long E_NMC = 5000l; // MC samplings to use for computing the energy
    const long G_NMC = 10000l; // MC samplings to use for computing the energy gradient
    double * energy = new double[4]; // energy
    double * d_energy = new double[4]; // energy error bar
    double * vp = new double[psi->getNVP()];


    cout << "-> ham1:    w = " << w1 << endl << endl;
    vmc = new VMC(psi, ham);

    // set an integration range, because the NN might be completely delocalized
    double ** irange = new double*[1];
    irange[0] = new double[2];
    irange[0][0] = -7.5;
    irange[0][1] = 7.5;
    vmc->getMCI()->setIRange(irange);


    cout << "   Starting energy:" << endl;
    vmc->computeVariationalEnergy(E_NMC, energy, d_energy);
    cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
    cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
    cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
    cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl;




    cout << "   Optimization . . ." << endl;
    vmc->conjugateGradientOptimization(E_NMC, G_NMC);
    cout << "   . . . Done!" << endl << endl;

    cout << "   Optimized energy:" << endl;
    vmc->computeVariationalEnergy(E_NMC, energy, d_energy);
    cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
    cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
    cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
    cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl << endl;


    // store in a .txt file the values of the optimised wf, so that it is possible to plot it
    cout << "Writing the plot file of the optimised wave function in plot_opt_wf.txt" << endl << endl;
    writePlotFile(psi->getBareFFNN(), base_input, input_i, output_i, min, max, npoints, "getOutput", "plot_opt_wf.txt");



    delete[] irange[0];
    delete[] irange;
    delete vmc;
    delete[] vp;
    delete[] d_energy;
    delete[] energy;
    delete ham;
    delete base_input;
    delete psi;
    delete ffnn;



    return 0;
}
