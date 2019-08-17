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
#include "nnvmc/ANNWaveFunction.hpp"
#include "nnvmc/DistanceFeedWrapper.hpp"
#include "qnets/templ/TemplNet.hpp"
#include "qnets/actf/TanSig.hpp"
#include "qnets/actf/Exp.hpp"
#include "sannifa/QTemplWrapper.hpp"


#include "../common/ExampleFunctions.hpp"

#include <cassert>

void checkDerivatives(vmc::WaveFunction &wf, const double re0[], double dx, double REL_TINY, double MIN_DIFF, bool verbose = false)
{
    const int ndim = wf.getTotalNDim();
    double protov[wf.getNProto()]; // holds temp values to compute wf value
    double vp0[wf.getNVP()];
    wf.getVP(vp0); // store initial vp

    for (int i = 0; i < ndim; ++i) {
        double re1[ndim];
        std::copy(re0, re0 + ndim, re1);
        wf.protoFunction(re0, protov);
        const double psi0 = wf.computeWFValue(protov);
        re1[i] = re0[i] + dx;
        wf.protoFunction(re1, protov);
        const double psir = wf.computeWFValue(protov);
        re1[i] = re0[i] - dx;
        wf.protoFunction(re1, protov);
        const double psil = wf.computeWFValue(protov);
        wf.computeAllDerivatives(re0);
        const double d1d_ana = wf.getD1DivByWF(i)*psi0;
        const double d1d_num = 0.5*(psir - psil)/dx;
        const double diffd1 = fabs(d1d_ana - d1d_num);
        const double rdiffd1 = d1d_num != 0. ? fabs(diffd1/d1d_num) : 0.;
        if (verbose) { std::cout << "Check dimension " << i << ":" << std::endl; }
        if (verbose) { std::cout << "d1: ana " << d1d_ana << " num " << d1d_num << " diff " << diffd1 << " reldiff " << rdiffd1 << std::endl; }
        //assert(rdiffd1 < REL_TINY || diffd1 < MIN_DIFF);
        const double d2d_ana = wf.getD2DivByWF(i)*psi0;
        const double d2d_num = (psir + psil - 2*psi0)/(dx*dx);
        const double diffd2 = fabs(d2d_ana - d2d_num);
        const double rdiffd2 = d2d_num != 0. ? fabs(diffd2/d2d_num) : 0.;
        if (verbose) { std::cout << "d2: ana " << d2d_ana << " num " << d2d_num << " diff " << diffd2 << " reldiff " << rdiffd2 << std::endl; }
        //assert(rdiffd2 < REL_TINY || diffd2 < MIN_DIFF);
    }
    for (int i = 0; i < wf.getNVP(); ++i) {
        wf.protoFunction(re0, protov);
        const double psi0 = wf.computeWFValue(protov);
        wf.computeAllDerivatives(re0);
        const double vd1d_ana = wf.getVD1DivByWF(i)*psi0;
        double vp1[wf.getNVP()];
        std::copy(vp0, vp0 + wf.getNVP(), vp1);
        vp1[i] += dx;
        wf.setVP(vp1);
        wf.protoFunction(re0, protov);
        const double psir = wf.computeWFValue(protov);
        const double vd1d_num = (psir - psi0)/dx;
        const double diffvd1 = fabs(vd1d_ana - vd1d_num);
        const double rdiffvd1 = vd1d_num != 0. ? fabs(diffvd1/vd1d_num) : 0.;
        if (verbose) { std::cout << "Check vp " << i << "( " << vp0[i] << " ):" << std::endl; }
        if (verbose) { std::cout << "vd1: ana " << vd1d_ana << " num " << vd1d_num << " diff " << diffvd1 << " reldiff " << rdiffvd1 << std::endl; }
        assert(rdiffvd1 < REL_TINY || diffvd1 < MIN_DIFF);
        wf.setVP(vp0); // restore original vp
    }
}

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
    using NetType = TemplNet<RealT, dconf, 5 /*1 e-e + 4 e-p distances*/, L1Type, L2Type, L3Type>;
    using WrapperType = QTemplWrapper<NetType>;

    // --- VMC optimization of the NNWF


    if (myrank == 0) { cout << endl << " - - - FFNN-WF FUNCTION OPTIMIZATION - - - " << endl << endl; }

    // Declare Hamiltonian for H2
    const double drp = 1.4;
    HydrogenMoleculeHamiltonian ham(drp);
    if (myrank == 0) { cout << "-> ham:    drp = " << drp << endl << endl; }


    const double rp[6]{-0.5*drp, 0., 0., 0.5*drp, 0., 0.};
    using DistNNType = DistanceFeedWrapper<WrapperType>;
    DistNNType ann(WrapperType(), 3, 2, 2, rp);

    static_assert(DistanceFeedWrapper<WrapperType>::calcNDists(2, 2) == 5, ""); // static check for correct number of distances

    // create random generator
    random_device rdev;
    mt19937_64 rgen;
    rgen = mt19937_64(rdev());
    rgen.seed(0);
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


    // Declare the trial wave function
    // setup the individual components

    ANNWaveFunction<DistNNType> psi_nn(3, 2, ann);

    const double re0[6]{-0.3, -0.2, 0.2,
                         0.24, 0.32, -0.4};
    checkDerivatives(psi_nn, re0, 0.00001, 0.001, 1.e-8, true);


    MolecularSigmaOrbital psi_orb1(drp, 0);
    MolecularSigmaOrbital psi_orb2(drp, 1);
    // and put them together via MultiComponentWaveFunction
    vmc::MultiComponentWaveFunction psi(3, 2, true);
    psi.addWaveFunction(&psi_nn);
    psi.addWaveFunction(&psi_orb1);
    psi.addWaveFunction(&psi_orb2);

    using namespace vmc;
    const long E_NMC = 100000l; // MC samplings to use for computing the initial/final energy
    const long G_NMC = 20000l; // MC samplings to use for computing the energy and gradient
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
    //minimizeEnergy<EnergyGradientTargetFunction>(vmc, adam, E_NMC, G_NMC, false, 0.001);
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
