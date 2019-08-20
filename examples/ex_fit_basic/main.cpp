#include <iostream>
#include <cmath>
#include <random>
#include <mpi.h>

#include "../common/ExampleFunctions.hpp"

#include "vmc/VMC.hpp"
#include "vmc/MPIVMC.hpp"
#include "nfm/Adam.hpp"
#include "nfm/LogManager.hpp"
#include "nnvmc/SimpleNNWF.hpp"
#include "qnets/templ/TemplNet.hpp"
#include "qnets/actf/Sigmoid.hpp"
#include "qnets/actf/Exp.hpp"
#include "sannifa/QTemplWrapper.hpp"


int main()
{
    using namespace std;
    using namespace templ;

    const int myrank = MPIVMC::Init(); // to allow use with MPI-compiled VMC library


    // --- Setup and fit the ANN

    // Setup TemplNet
    constexpr auto dconf = DerivConfig::D12_VD1; // allow necessary derivatives (for VMC later)
    constexpr DynamicDFlags dflags0(DerivConfig::VD1); // for fitting we need only vd1
    using RealT = double;

    const int HIDDENLAYERSIZE = 12;
    using L1Type = LayerConfig<HIDDENLAYERSIZE, actf::Sigmoid>;
    using L2Type = LayerConfig<1, actf::Exp>;
    using NetType = TemplNet<RealT, dconf, 1, 1, L1Type, L2Type>;
    QTemplWrapper<NetType> ann(dflags0);

    const int nvpar = ann.getNVariationalParameters();
    double vp[nvpar];

    // we do the initialization and subsequent fitting on thread 0 only
    cout << "myrank: " << myrank << endl;
    if (myrank == 0) {
        // setup random generator
        random_device rdev;
        mt19937_64 rgen;
        rgen = mt19937_64(rdev());
        auto rd = normal_distribution<double>(0, sqrt(1./HIDDENLAYERSIZE));

        // setup initial weights
        int ivp_l1 = 0; // l1 weights running index
        int ivp_l2 = HIDDENLAYERSIZE*2; // l2 weights running index
        vp[ivp_l2] = 0.; // l2 bias init to 0
        for (int i = 0; i < HIDDENLAYERSIZE; ++i) {
            vp[ivp_l1] = 0.; // l1 bias init to 0
            vp[ivp_l1 + 1] = rd(rgen); // l1-l0 weights
            ivp_l1 += 2;
            vp[ivp_l2 + 1] = rd(rgen); // l2-l1 weights
            ++ivp_l2;
        }
        cout << "Init vp: ";
        for (double v : vp) { cout << v << " "; }
        cout << endl;

        ann.setVariationalParameters(vp);

        cout << endl << " - - - NNWF FITTING - - - " << endl << endl;

        // generate some gaussian data for fit
        const int ndata = 2000;
        rd = normal_distribution<double>(0, 2.); // sigma 2.
        double xdata[ndata], ydata[ndata];

        for (int i = 0; i < ndata; ++i) {
            xdata[i] = rd(rgen);
            ydata[i] = exp(-0.5*xdata[i]*xdata[i]);
            // if (myrank == 0) { cout << "y( " << xdata[i] << " ) = " << ydata[i] << endl; }
        }

        // setup fit cost
        NNFitCost fit_cost(&ann, ndata, xdata, ydata);

        // Setup Adam
        nfm::Adam adam(ann.getNVariationalParameters(), false /*disable averaging here*/);
        adam.setAlpha(0.01);
        adam.setBeta1(0.5);
        adam.setBeta2(0.9);
        adam.setMaxNConstValues(20);
        adam.setMaxNIterations(500);

        const double ftol_fit = 0.00002;
        nfm::NoisyIOPair res(ann.getNVariationalParameters()); // variable to hold optimization results
        std::copy(vp, vp + ann.getNVariationalParameters(), res.x.begin()); // copy initial pars to res
        res.f = {-1., 0.}; // set initial residual to negative value

        while (res.f < 0. || res.f > ftol_fit) { // repeat fitting until residual below tolerance
            res = adam.findMin(fit_cost, res.x.data());
            cout << "Fit residual: R = " << res.f.val << " +- " << res.f.err << endl;
        }

        for (double x = -5.; x <= 5.; x += 0.5) {
            double xv[1]{x};
            ann.evaluate(xv);
            cout << "P(" << x << ") = " << ann.getOutput(0) << endl;
        }
    }

    // broadcast fit result
    if (myrank == 0) { ann.getVariationalParameters(vp); }
    MPI_Bcast(vp, nvpar, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    ann.setVariationalParameters(vp);

    // --- Compute VMC energy of the fitted NNWF

    // Declare the trial wave functions
    ann.enableFirstDerivative();
    ann.enableSecondDerivative();
    SimpleNNWF<QTemplWrapper<NetType>> psi(1, 1, ann);

    // Declare an Hamiltonian
    // We use the harmonic oscillator with w=1 and w=2
    const double w1 = 1.;
    HarmonicOscillator1D1P ham(w1);


    if (myrank == 0) { cout << endl << " - - - NNWF ENERGY COMPUTATION - - - " << endl << endl; }

    using namespace vmc;
    const long E_NMC = 1000000l; // MC samplings to use for computing the initial/final energy
    double energy[4]; // energy
    double d_energy[4]; // energy error bar

    if (myrank == 0) { cout << "-> ham1:    w = " << w1 << endl << endl; }
    VMC vmc(psi, ham); // VMC object used for energy computation

    vmc.computeEnergy(E_NMC, energy, d_energy);
    if (myrank == 0) {
        cout << "   Resulting energy:" << endl;
        cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
        cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
        cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
        cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl;
    }

    MPIVMC::Finalize();

    return 0;
}
