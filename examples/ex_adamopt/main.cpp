#include <iostream>
#include <cmath>
#include <stdexcept>

#include "../common/ExampleFunctions.hpp"

#include "vmc/VMC.hpp"
#include "vmc/MPIVMC.hpp"
#include "vmc/EnergyMinimization.hpp"
#include "nfm/Adam.hpp"
#include "nfm/LogManager.hpp"
#include "nnvmc/ANNWaveFunction.hpp"
#include "qnets/poly/FeedForwardNeuralNetwork.hpp"
#include "qnets/poly/io/PrintUtilities.hpp"
#include "sannifa/QPolyWrapper.hpp"


int main()
{
    using namespace std;

    MPIVMC::Init(); // to avoid error with MPI-compiled VMC library

    // construct NNWF
    const int HIDDENLAYERSIZE = 13; // including offset
    const int NHIDDENLAYERS = 1;
    FeedForwardNeuralNetwork ffnn(2, HIDDENLAYERSIZE, 2);
    for (int i = 0; i < NHIDDENLAYERS - 1; ++i) {
        ffnn.pushHiddenLayer(HIDDENLAYERSIZE);
    }

    // change hidden layer ACTFs to tansigmoid
    for (int il = 0; il < ffnn.getNNeuralLayers() - 1; ++il) {
        for (int iu = 0; iu < ffnn.getNNLayer(il)->getNNeuralUnits(); ++iu) {
            ffnn.getNNLayer(il)->getNNUnit(iu)->setActivationFunction(new TanSigmoidActivationFunction());
        }
    }
    ffnn.getOutputLayer()->getNNUnit(0)->setActivationFunction(new ExponentialActivationFunction());

    ffnn.connectFFNN();
    ffnn.assignVariationalParameters();

    QPolyWrapper nn_wrapper(ffnn);
    nn_wrapper.enableFirstDerivative();
    nn_wrapper.enableSecondDerivative();
    nn_wrapper.enableVariationalFirstDerivative();

    // Declare the trial wave functions
    using NNWFType = ANNWaveFunction<QPolyWrapper>;
    NNWFType psi(1, 1, nn_wrapper);

    // Declare an Hamiltonian
    // We use the harmonic oscillator with w=1 and w=2
    const double w1 = 1.;
    HarmonicOscillator1D1P ham(w1);


    cout << endl << " - - - FFNN-WF FUNCTION OPTIMIZATION - - - " << endl << endl;

    using namespace vmc;
    const long E_NMC = 100000l; // MC samplings to use for computing the energy (init/final)
    const long G_NMC = 20000l; // MC samplings to use for computing the energy and gradient
    double energy[4]; // energy
    double d_energy[4]; // energy error bar

    cout << "-> ham1:    w = " << w1 << endl << endl;
    VMC vmc(psi, ham); // VMC object used for energy/optimization


    // Saving a (non-const) reference to the internally used ffnn,
    // to evaluate it for plotting without making copies of it
    // Calling non-const methods on this reference should be
    // done with thought. Just evaluating the NN is OK though.
    auto &used_ffnn = const_cast<FeedForwardNeuralNetwork &>(dynamic_cast<NNWFType &>(vmc.getWF()).getANN().getBareFFNN());

    // Store in a .txt file the values of the initial wf, so that it is possible to plot it
    cout << "Writing the plot file of the initial wave function in plot_init_wf.txt" << endl << endl;
    double base_input[used_ffnn.getNInput()]; // no need to set it, since it is 1-dim
    const int input_i = 0;
    const int output_i = 0;
    const double min = -5.;
    const double max = 5.;
    const int npoints = 500;
    writePlotFile(&used_ffnn, base_input, input_i, output_i, min, max, npoints, "getOutput", "plot_init_wf.txt");
    printFFNNStructure(&used_ffnn);

    // set an integration range, because the NN might be completely delocalized
    vmc.getMCI().setIRange(-7.5, 7.5);

    // set fixed number of decorrelation steps
    vmc.getMCI().setNdecorrelationSteps(1000);


    cout << "   Starting energy:" << endl;
    vmc.computeEnergy(E_NMC, energy, d_energy);
    cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
    cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
    cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
    cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl;


    // set logging
    nfm::LogManager::setLoggingOn();

    // -- Setup Adam optimization
    nfm::Adam adam(vmc.getNVP(), true /* use averaging to obtain final result */);
    adam.setAlpha(0.02);
    adam.setBeta1(0.5);
    adam.setBeta2(0.9);
    adam.setMaxNConstValues(20); // stop after 20 constant values (within error)
    adam.setMaxNIterations(200); // limit to 200 iterations


    cout << "   Optimization . . ." << endl;
    minimizeEnergy<EnergyGradientTargetFunction>(vmc, adam, E_NMC, G_NMC);
    cout << "   . . . Done!" << endl << endl;

    cout << "   Optimized energy:" << endl;
    vmc.computeEnergy(E_NMC, energy, d_energy);
    cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
    cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
    cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
    cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl << endl;


    // store in a .txt file the values of the optimised wf, so that it is possible to plot it
    cout << "Writing the plot file of the optimised wave function in plot_opt_wf.txt" << endl << endl;
    writePlotFile(&used_ffnn, base_input, input_i, output_i, min, max, npoints, "getOutput", "plot_opt_wf.txt");


    MPIVMC::Finalize();

    return 0;
}