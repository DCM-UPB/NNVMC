#include <iostream>
#include <cmath>
#include <stdexcept>

#include "vmc/Hamiltonian.hpp"
#include "vmc/VMC.hpp"
#include "vmc/MPIVMC.hpp"
#include "ffnn/net/FeedForwardNeuralNetwork.hpp"
#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/io/PrintUtilities.hpp"
#include "nnvmc/FFNNWaveFunction.hpp"


/*
Hamiltonian describing a n-particle harmonic oscillator:
   H  =  Sum_i^n( p_i^2 ) /2m  +  1/2 * w^2 * Sum_i^n( x_i^2 )
*/
class HarmonicOscillatorNDNP: public Hamiltonian{

protected:
   int _n;
   double _wsqh;

public:
   HarmonicOscillatorNDNP(const double w, FFNNWaveFunction * wf):
     Hamiltonian(wf->getNSpaceDim() /*num space dimensions*/, wf->getNPart() /*num particles*/, wf) {
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

int main(){
   using namespace std;

   MPIVMC::Init();

   const int NSPACEDIM = 1;
   const int NHIDDENLAYERS = 1;
   const int HIDDENLAYERSIZE[NHIDDENLAYERS] = {10};
   const int NPARTICLES = 1;
   FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(NSPACEDIM*NPARTICLES + 1, HIDDENLAYERSIZE[0], 2);
   //FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork("nn_in.txt");
   for (int i=1; i<NHIDDENLAYERS; ++i){
      ffnn->pushHiddenLayer(HIDDENLAYERSIZE[i]);
   }
   ffnn->connectFFNN();
   ffnn->assignVariationalParameters();

   //Set ACTFs for hidden units
   for (int i=0; i<NHIDDENLAYERS; ++i) {
       for (int j=0; j<HIDDENLAYERSIZE[i]-1; ++j) {
           ffnn->getNNLayer(i)->getNNUnit(j)->setActivationFunction(std_actf::provideActivationFunction("TANS"));
       }
   }

   cout << "Created FFNN with " << NHIDDENLAYERS << " hidden layer(s) of ";
   for(int i=0; i<NHIDDENLAYERS; ++i) {cout << HIDDENLAYERSIZE[i] << ", ";} 
   cout << " units each." << endl << endl;

   // Declare the trial wave functions
   FFNNWaveFunction * psi = new FFNNWaveFunction(NSPACEDIM, NPARTICLES, ffnn, false, false, false);

   // Store in two files the the initial wf, one for plotting, and for recovering it as nn
   cout << "Writing the plot file of the initial wave function in (plot_)init_wf.txt" << endl << endl;
   double * base_input = new double[psi->getBareFFNN()->getNInput()];
   for (int i=0; i<psi->getBareFFNN()->getNInput(); ++i) base_input[i] = 0.;
   const double min = -5.;
   const double max = 5.;
   const int npoints = 100;
   writePlotFile(psi->getBareFFNN(), base_input, 0, 0, min, max, npoints, "getOutput", "plot_init_wf.txt");
   psi->getBareFFNN()->storeOnFile("wf_init.txt");


   // Declare an Hamiltonian
   // We use the harmonic oscillator with w=1
   double w1 = 1.0;

   HarmonicOscillatorNDNP * ham = new HarmonicOscillatorNDNP(w1, psi);

   cout << endl << " - - - FFNN-WF FUNCTION OPTIMIZATION - - - " << endl << endl;

   VMC * vmc; // VMC object we will resuse
   const long E_NMC = 40000l; // MC samplings to use for computing the energy
   cout << "E_NMC = " << E_NMC << endl << endl;
   double energy[4]; // energy
   double d_energy[4]; // energy error bar

   cout << "-> ham1:    w = " << w1 << endl << endl;
   vmc = new VMC(psi, ham);

   // set an integration range, because the NN might be completely delocalized
   double ** irange = new double*[NSPACEDIM*NPARTICLES];
   for (int i=0; i<NSPACEDIM*NPARTICLES; ++i) {irange[i] = new double[2]; irange[i][0] = -5.; irange[i][1] = 5.;}
   cout << "Integration range: " << irange[0][0] << "   <->   " << irange[0][1] << endl << endl;
   vmc->getMCI()->setIRange(irange);

   // for better performance:
   vmc->getMCI()->setNdecorrelationSteps(1000);
   vmc->getMCI()->setNfindMRT2steps(10);

   cout << "   Starting energy:" << endl;
   vmc->computeVariationalEnergy(E_NMC, energy, d_energy);
   cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
   cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
   cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
   cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl;

   cout << "   Optimization . . ." << endl;
   vmc->nmsimplexOptimization(E_NMC, 1.0, 0.1, 0.005);
   cout << "   . . . Done!" << endl << endl;

   cout << "   Optimized energy:" << endl;
   vmc->computeVariationalEnergy(E_NMC, energy, d_energy);
   cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
   cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
   cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
   cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl << endl;


   // Store in two files the the initial wf, one for plotting, and for recovering it as nn

   cout << "Writing the plot file of the optimised wave function in (plot_)opt_wf.txt" << endl << endl;
   writePlotFile(psi->getBareFFNN(), base_input, 0, 0, min, max, npoints, "getOutput", "plot_opt_wf.txt");
   psi->getBareFFNN()->storeOnFile("wf_opt.txt");


   delete[] irange[0];
   delete[] irange;
   delete vmc;
   delete ham;
   delete base_input;
   delete psi;
   delete ffnn;

   MPIVMC::Finalize();

   return 0;
}
