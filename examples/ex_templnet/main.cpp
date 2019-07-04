#include <iostream>
#include <cmath>
#include <random>

#include "../common/ExampleFunctions.hpp"

#include "vmc/VMC.hpp"
#include "vmc/MPIVMC.hpp"
#include "vmc/EnergyMinimization.hpp"
#include "nfm/DynamicDescent.hpp"
#include "nfm/Adam.hpp"
#include "nfm/LogManager.hpp"
#include "nnvmc/ANNWaveFunction.hpp"
#include "qnets/templ/TemplNet.hpp"
#include "qnets/actf/Sigmoid.hpp"
#include "qnets/actf/Exp.hpp"
#include "qnets/actf/ReLU.hpp"
#include "sannifa/QTemplWrapper.hpp"


int main()
{
    using namespace std;
    using namespace templ;

    MPIVMC::Init(); // to avoid error with MPI-compiled VMC library


    // --- Setup and pre-fit the ANN

    // Setup TemplNet
    constexpr auto dconf = DerivConfig::D12_VD1; // enable necessary derivatives
    constexpr DynamicDFlags dflags0(DerivConfig::D12_VD1);
    using RealT = double;

    const int HIDDENLAYERSIZE = 8;
    using L1Type = LayerConfig<HIDDENLAYERSIZE, actf::Sigmoid>;
    using L2Type = LayerConfig<HIDDENLAYERSIZE, actf::Sigmoid>;
    using L3Type = LayerConfig<1, actf::Exp>;
    using NetType = TemplNet<RealT, dconf, 1, L1Type, L2Type, L3Type>;
    QTemplWrapper<NetType> ann;

    // generate some random betas
    random_device rdev;
    mt19937_64 rgen;
    rgen = mt19937_64(rdev());
    auto rd = normal_distribution<double>(0.01); // uniform with variance 1

    double vp[ann.getNVariationalParameters()];
    for (auto & v : vp) { v = rd(rgen); }
    ann.setVariationalParameters(vp);

    /*
    // generate some gaussian data for pre-fit
    const int ndata = 10000;
    const double rdscale = 500.;
    double xdata[ndata], ydata[ndata];

    for (int i = 0; i < ndata; ++i) {
        xdata[i] = rdscale*rd(rgen);
        ydata[i] = exp(-xdata[i]*xdata[i]);
    }

    // setup fitting
    NNFitCost fit_cost(&ann, ndata, xdata, ydata);
    */

    /*nfm::ConjGrad cg(ann.getNVariationalParameters());
    cg.setStepSize(0.01);
    cg.setBackStep(0.005);
    cg.setMaxNBracket(10);
    cg.setEpsX(0.);
    cg.setEpsF(0.);
    cg.setMaxNIterations(50);
    cg.useConjGradFR(); // in the noise case Fletcher-Reeves seems to do best
    nfm::LogManager::setLoggingOn(true); // verbose logging
    for (int i = 0; i < ann.getNVariationalParameters(); ++i) { cg.setX(vp); }

    cg.findMin(fit_cost); // do the fit
    */

    /*
    nfm::DynamicDescent dd(ann.getNVariationalParameters());
    dd.setStepSize(0.001);
    nfm::LogManager::setLoggingOn();
    dd.setX(vp);
    dd.findMin(fit_cost); // do the fit
    */

    // --- VMC optimization of the NNWF

    // Declare the trial wave functions
    ANNWaveFunction<QTemplWrapper<NetType>> psi(1, 1, ann);

    // Declare an Hamiltonian
    // We use the harmonic oscillator with w=1 and w=2
    const double w1 = 1.;
    HarmonicOscillator1D1P ham(w1);


    cout << endl << " - - - FFNN-WF FUNCTION OPTIMIZATION - - - " << endl << endl;

    using namespace vmc;
    const long E_NMC = 100000l; // MC samplings to use for computing the initial/final energy
    const long G_NMC = 20000l; // MC samplings to use for computing the energy and gradient
    double energy[4]; // energy
    double d_energy[4]; // energy error bar

    cout << "-> ham1:    w = " << w1 << endl << endl;
    VMC vmc(psi, ham); // VMC object used for energy/optimization

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


    // -- Setup Adam optimization
    nfm::Adam adam(psi.getNVP(), true /* use averaging to obtian final result */);
    adam.setAlpha(0.015);
    adam.setBeta1(0.5);
    adam.setBeta2(0.9);
    adam.setMaxNConstValues(20); // stop after 10 constant values (within error)
    adam.setMaxNIterations(200); // limit to 200 iterations
    nfm::LogManager::setLoggingOn(); // normal logging


    cout << "   Optimization . . ." << endl;
    minimizeEnergy<EnergyGradientTargetFunction>(vmc, adam, E_NMC, G_NMC);
    cout << "   . . . Done!" << endl << endl;

    cout << "   Optimized energy:" << endl;
    vmc.computeEnergy(E_NMC, energy, d_energy);
    cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
    cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
    cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
    cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl << endl;


    MPIVMC::Finalize();

    return 0;
}
