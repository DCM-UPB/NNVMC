#include <iostream>
#include <cmath>
#include <random>
#include <mpi.h>

#include "vmc/MultiComponentWaveFunction.hpp"
#include "vmc/VMC.hpp"
#include "vmc/MPIVMC.hpp"
#include "vmc/EnergyMinimization.hpp"
#include "nfm/Adam.hpp"
#include "nfm/LogManager.hpp"
#include "nnvmc/SimpleNNWF.hpp"
#include "qnets/templ/TemplNet.hpp"
#include "qnets/actf/TanSig.hpp"
#include "qnets/actf/Exp.hpp"
#include "sannifa/QTemplWrapper.hpp"


#include "../common/ExampleFunctions.hpp"


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
    using L2Type = LayerConfig<HIDDENLAYERSIZE, actf::TanSig>;
    using L3Type = LayerConfig<1, actf::Exp>;
    using NetType = TemplNet<RealT, dconf, 6, 6 /*6 electronic coordinates*/, L1Type, L2Type, L3Type>;
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
        for (double &v : vp) { v = rd(rgen); }
    }
    // broadcast the chosen weights to all threads
    MPI_Bcast(vp, nvpar, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    ann.setVariationalParameters(vp);

    // --- VMC optimization of the NNWF


    if (myrank == 0) { cout << endl << " - - - FFNN-WF FUNCTION OPTIMIZATION - - - " << endl << endl; }

    // Declare Hamiltonian for H2
    const double drp = 1.4;
    HydrogenMoleculeHamiltonian ham(drp);
    if (myrank == 0) { cout << "-> ham:    drp = " << drp << endl << endl; }

    // Declare the trial wave function
    // setup the individual components
    SimpleNNWF<QTemplWrapper<NetType>> psi_nn(3, 2, ann);
    MolecularSigmaOrbital psi_orb1(drp, 0);
    MolecularSigmaOrbital psi_orb2(drp, 1);
    // and put them together via MultiComponentWaveFunction
    vmc::MultiComponentWaveFunction psi(3, 2, true);
    psi.addWaveFunction(&psi_nn);
    psi.addWaveFunction(&psi_orb1);
    psi.addWaveFunction(&psi_orb2);

    using namespace vmc;
    const long E_NMC = 1048576; // MC samplings to use for computing the initial/final energy
    const long G_NMC = 32768; // MC samplings to use for computing the energy and gradient
    double energy[4]; // energy
    double d_energy[4]; // energy error bar

    VMC vmc(psi, ham); // VMC object used for energy/optimization

    // set an integration range, because the NN might be completely delocalized
    vmc.getMCI().setIRange(-10., 10.);

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
    adam.setAlpha(0.01);
    adam.setBeta1(0.9);
    adam.setBeta2(0.9);
    adam.setMaxNConstValues(25); // stop after 25 constant values (within error)
    adam.setMaxNIterations(250); // limit to 250 iterations
    if (myrank == 0) { nfm::LogManager::setLoggingOn(); } // normal logging on thread 0

    // optimize the NNWF
    if (myrank == 0) { cout << "   Optimization . . ." << endl; }
    minimizeEnergy<EnergyGradientTargetFunction>(vmc, adam, E_NMC, G_NMC, false, 0.001);
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
