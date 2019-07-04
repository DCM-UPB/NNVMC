#include <iostream>
#include <cmath>
#include <random>

#include "../common/ExampleFunctions.hpp"

#include "vmc/VMC.hpp"
#include "vmc/MPIVMC.hpp"
#include "nfm/Adam.hpp"
#include "nfm/LogManager.hpp"
#include "nnvmc/ANNWaveFunction.hpp"
#include "qnets/templ/TemplNet.hpp"
#include "qnets/actf/Sigmoid.hpp"
#include "qnets/actf/Exp.hpp"
#include "sannifa/QTemplWrapper.hpp"


int main()
{
    using namespace std;
    using namespace templ;

    MPIVMC::Init(); // to avoid error with MPI-compiled VMC library


    // --- Setup and fit the ANN

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
    auto rd = normal_distribution<double>(0, sqrt(1./HIDDENLAYERSIZE));

    double vp[ann.getNVariationalParameters()];
    for (auto &v : vp) { v = rd(rgen); }
    ann.setVariationalParameters(vp);

    cout << endl << " - - - NNWF FITTING - - - " << endl << endl;

    // generate some gaussian data for fit
    const int ndata = 2000;
    rd = normal_distribution<double>(0, 2.); // sigma 2.
    double xdata[ndata], ydata[ndata];

    for (int i = 0; i < ndata; ++i) {
        xdata[i] = rd(rgen);
        ydata[i] = exp(-0.5*xdata[i]*xdata[i]);
        cout << "y( " << xdata[i] << " ) = " << ydata[i] << endl;
    }

    // setup fitting
    NNFitCost fit_cost(&ann, ndata, xdata, ydata);

    nfm::Adam adam(ann.getNVariationalParameters());
    adam.setAlpha(0.01);
    adam.setBeta1(0.75);
    adam.setBeta2(0.95);
    adam.setMaxNConstValues(20);
    adam.setMaxNIterations(500);
    adam.findMin(fit_cost, vp);

    for (double x = -5.; x <= 5.; x += 0.5) {
        double xv[1]{x};
        ann.evaluate(xv);
        cout << "P(" << x << ") = " << ann.getOutput(0) << endl;
    }

    // --- Compute VMC energy of the fitted NNWF

    // Declare the trial wave functions
    ANNWaveFunction<QTemplWrapper<NetType>> psi(1, 1, ann);

    // Declare an Hamiltonian
    // We use the harmonic oscillator with w=1 and w=2
    const double w1 = 1.;
    HarmonicOscillator1D1P ham(w1);


    cout << endl << " - - - NNWF ENERGY COMPUTATION - - - " << endl << endl;

    using namespace vmc;
    const long E_NMC = 100000l; // MC samplings to use for computing the initial/final energy
    double energy[4]; // energy
    double d_energy[4]; // energy error bar

    cout << "-> ham1:    w = " << w1 << endl << endl;
    VMC vmc(psi, ham); // VMC object used for energy/optimization


    cout << "   Resulting energy:" << endl;
    vmc.computeEnergy(E_NMC, energy, d_energy);
    cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
    cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
    cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
    cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl;


    MPIVMC::Finalize();

    return 0;
}
