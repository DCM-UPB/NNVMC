#include <iostream>
#include <cmath>
#include <random>
#include <mpi.h>

#include "vmc/Hamiltonian.hpp"
#include "vmc/WaveFunction.hpp"
#include "vmc/MultiComponentWaveFunction.hpp"
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

#include <iostream>
#include <cmath>

// helper struct and function to determine the two e-p distances per electron
struct EPDistances
{
    double v[2];
};

// compute distances, given that protons are aligned along x-axis
EPDistances calcEPDistances(const double r[2], double drp)
{
    const double hdrp = 0.5*drp;
    const double hdrp2 = hdrp*hdrp;
    EPDistances dists{};

    const double yzdist = r[1]*r[1] + r[2]*r[2]; // y/z directional parts of e-p distance
    const double xdist_1 = r[0]*r[0] + hdrp2;
    const double xdist_2 = 2.*r[0]*hdrp;

    dists.v[0] = sqrt(xdist_1 + xdist_2 + yzdist);
    dists.v[1] = sqrt(xdist_1 - xdist_2 + yzdist);

    return dists;
}

/*
  Electronic Hamiltonian describing a single H2 molecule aligned to x-axis
*/
class HydrogenMoleculeHamiltonian: public vmc::Hamiltonian
{
protected:
    const double _drp; // p-p distance
    const double e0; // protonic coulomb energy

    mci::ObservableFunctionInterface * _clone() const final
    {
        return new HydrogenMoleculeHamiltonian(_drp);
    }
public:
    explicit HydrogenMoleculeHamiltonian(double drp):
            vmc::Hamiltonian(3 /*num space dimensions*/, 2 /*num particles*/),
            _drp(drp), e0(1./drp) {}

    // potential energy
    double localPotentialEnergy(const double r[]) final
    {
        double pot = 0.;
        // add e-p coulomb terms
        for (int i = 0; i < 2; ++i) {
            const EPDistances dists = calcEPDistances(r + i*3, _drp);
            pot -= 1./dists.v[0];
            pot -= 1./dists.v[1];
        }
        // add e-e coulomb terms
        double dist = 0.;
        for (int k = 0; k < 3; ++k) {
            dist += (r[k] - r[k + 3])*(r[k] - r[k + 3]);
        }
        pot += 1./sqrt(dist);

        return pot + e0; // add p-p coulomb term and return
    }
};

/*
  HydrogenMolecule 1P sigma orbitals
*/
class MolecularSigmaOrbital: public vmc::WaveFunction
{
protected:
    const double _drp;
    const int _pindex;

    mci::SamplingFunctionInterface * _clone() const final
    {
        return new MolecularSigmaOrbital(_drp, _pindex, this->hasVD1());
    }

public:
    MolecularSigmaOrbital(double drp, int part_index /* for which of the particles is the orbital */, bool flag_vd1 = true):
            WaveFunction(3 /*num space dimensions*/, 2 /*num particles*/, 1 /*num wf components*/, 0 /*num variational parameters*/, flag_vd1 /*VD1*/, false /*D1VD1*/, false /*D2VD1*/),
            _drp(drp), _pindex(part_index) {}

    void setVP(const double in[]) final {}
    void getVP(double out[]) const final {}

    void protoFunction(const double in[], double out[]) final
    {
            EPDistances dists = calcEPDistances(in + _pindex*3, _drp);
            out[0] = 0.5*(exp(-dists.v[0]) + exp(-dists.v[1]));
    }

    double acceptanceFunction(const double protoold[], const double protonew[]) const final
    {
        if (protoold[0] == 0.) {
            return (protonew[0] != 0.) ? 1. : 0.;
        }
        return (protonew[0]*protonew[0])/(protoold[0]*protoold[0]);
    }

    void computeAllDerivatives(const double in[]) final
    {
        double wfval[2];
        const int pidx = _pindex*3; // particle index offset
        for (int k = 0; k < 6; ++k) { // set all deriv elements to 0
            _setD1DivByWF(k, 0.);
            _setD2DivByWF(k, 0.);
        }
        EPDistances dists = calcEPDistances(in + pidx, _drp);
        for (int j = 0; j < 2; ++j) {
            const double dist = dists.v[j];
            const double dist2 = dist*dist;
            const double dist3 = dist2*dist;
            const double sig_hdrp = ( j == 0) ? -0.5*_drp : 0.5*_drp;
            wfval[j] = exp(-dist);

            // compute x elements of derivs
            _setD1DivByWF(pidx, getD1DivByWF(pidx) - (in[pidx] - sig_hdrp)/dist*wfval[j]);
            _setD2DivByWF(pidx, getD2DivByWF(pidx) + ((in[pidx] - sig_hdrp)*(in[pidx] - sig_hdrp)*(1./dist2 + 1./dist3) - 1./dist)*wfval[j]);

            // compute other directions
            for (int k = 1; k < 3; ++k) {
                _setD1DivByWF(pidx + k, getD1DivByWF(pidx + k) - (in[pidx + k])/dist*wfval[j]);
                _setD2DivByWF(pidx + k, getD2DivByWF(pidx + k) + ((in[pidx + k]*in[pidx + k])*(1./dist2 + 1./dist3) - 1./dist)*wfval[j]);
            }
        }
        // add final terms
        const double wfval_full = wfval[0] + wfval[1];
        for (int k = 0; k < 3; ++k) {
            _setD1DivByWF(pidx + k, getD1DivByWF(pidx + k)/wfval_full);
            _setD2DivByWF(pidx + k, getD2DivByWF(pidx + k)/wfval_full);
        }
    }

    double computeWFValue(const double protovalues[]) const final
    {
        return protovalues[0];
    }
};


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
    using NetType = TemplNet<RealT, dconf, 6 /*6 electronic coordinates*/, L1Type, L2Type, L3Type>;
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
    ANNWaveFunction<QTemplWrapper<NetType>> psi_nn(3, 2, ann);
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
