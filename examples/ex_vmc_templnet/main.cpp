#include <iostream>
#include <cmath>
#include <random>
#include <mpi.h>

#include "../common/ExampleFunctions.hpp"

#include "vmc/VMC.hpp"
#include "vmc/MPIVMC.hpp"
#include "vmc/EnergyMinimization.hpp"
#include "nfm/Adam.hpp"
#include "nfm/LogManager.hpp"
#include "nnvmc/ANNWaveFunction.hpp"
#include "qnets/templ/TemplNet.hpp"
#include "qnets/actf/TanSig.hpp"
#include "qnets/actf/Exp.hpp"
#include "sannifa/QTemplWrapper.hpp"


int main()
{
    using namespace std;
    using namespace templ;

    const int myrank = MPIVMC::Init(); // to allow use with MPI-compiled VMC library


    // --- Setup the ANN

    // Setup TemplNet
    constexpr auto dconf = DerivConfig::D12_VD1; // configure necessary derivatives
    using RealT = double;

    const int HIDDENLAYERSIZE = 12; // excluding (!) offset "unit"
    using L1Type = LayerConfig<HIDDENLAYERSIZE, actf::TanSig>;
    using L2Type = LayerConfig<1, actf::Exp>;
    using NetType = TemplNet<RealT, dconf, 1, L1Type, L2Type>;
    QTemplWrapper<NetType> ann;

    // create random generator
    random_device rdev;
    mt19937_64 rgen;
    rgen = mt19937_64(rdev());
    auto rd = normal_distribution<double>(0, sqrt(1./HIDDENLAYERSIZE)); // rule-of-thumb sigma

    // setup initial weights
    const int nvpar = ann.getNVariationalParameters();
    double vp[nvpar];
    if (myrank == 0) { // we need to make sure that all threads use the same initial weights
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
    }
    // broadcast the chosen weights to all threads
    MPI_Bcast(vp, nvpar, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    ann.setVariationalParameters(vp);

    // --- VMC optimization of the NNWF

    // Declare the trial wave functions
    ANNWaveFunction<QTemplWrapper<NetType>> psi(1, 1, ann);

    // Declare an Hamiltonian
    // We use the harmonic oscillator with w=1 and w=2
    const double w1 = 1.;
    HarmonicOscillator1D1P ham(w1);


    if (myrank == 0) { cout << endl << " - - - FFNN-WF FUNCTION OPTIMIZATION - - - " << endl << endl; }

    using namespace vmc;
    const long E_NMC = 100000l; // MC samplings to use for computing the initial/final energy
    const long G_NMC = 20000l; // MC samplings to use for computing the energy and gradient
    double energy[4]; // energy
    double d_energy[4]; // energy error bar

    if (myrank == 0) { cout << "-> ham1:    w = " << w1 << endl << endl; }
    VMC vmc(psi, ham); // VMC object used for energy/optimization

    // set an integration range, because the NN might be completely delocalized
    vmc.getMCI().setIRange(-7.5, 7.5);

    // set fixed number of decorrelation steps
    vmc.getMCI().setNdecorrelationSteps(1000);

    // compute initial energy
    vmc.computeEnergy(E_NMC, energy, d_energy);
    if (myrank == 0) {
        cout << "   Starting energy:" << endl;
        cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
        cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
        cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
        cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl;
    }

    // -- Setup Adam optimization
    nfm::Adam adam(psi.getNVP(), true /* use averaging to obtian final result */);
    adam.setAlpha(0.02);
    adam.setBeta1(0.5);
    adam.setBeta2(0.9);
    adam.setMaxNConstValues(20); // stop after 10 constant values (within error)
    adam.setMaxNIterations(200); // limit to 200 iterations
    if (myrank == 0) { nfm::LogManager::setLoggingOn(); } // normal logging on thread 0

    // optimize the NNWF
    if (myrank == 0) { cout << "   Optimization . . ." << endl; }
    minimizeEnergy<EnergyGradientTargetFunction>(vmc, adam, E_NMC, G_NMC);
    if (myrank == 0) { cout << "   . . . Done!" << endl << endl; }

    // compute final energy
    vmc.computeEnergy(E_NMC, energy, d_energy);
    if (myrank == 0) {
        cout << "   Optimized energy:" << endl;
        cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
        cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
        cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
        cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl << endl;
    }

    MPIVMC::Finalize();

    return 0;
}
