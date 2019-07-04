#include "vmc/WaveFunction.hpp"
#include "vmc/Hamiltonian.hpp"
#include "vmc/VMC.hpp"
#include "vmc/MPIVMC.hpp"
#include "qnets/poly/FeedForwardNeuralNetwork.hpp"
#include "qnets/poly/actf/GaussianActivationFunction.hpp"
#include "qnets/poly/actf/IdentityActivationFunction.hpp"
#include "qnets/poly/io/PrintUtilities.hpp"
#include "sannifa/QPolyWrapper.hpp"
#include "nnvmc/ANNWaveFunction.hpp"

#include <iostream>
#include <assert.h>
#include <math.h>
#include <stdexcept>


/*

  In this test we check that the ANNWaveFunction works as expected,
  using the Sannifa-wrapped QNets/PolyNet as backend.
  In particular we want to be sure that all the derivatives are computed properly.

  To accomplish this, we are following this approach.
  We build a NN with the following structure:

  id_            id_            id_

  id_     p1     gss     +0.00  id_
  p2             +1.00

  where p1 nad p2 are two variational parameters.
  This means that the NN is the following function:

  nn(x) = exp( - ( p1 + p2*x )^2 )

  i.e. a gaussian. We write a WaveFunction that has the same structure:

  gauss(x) = exp( - b ( x - a )^2 )

  There is therefore the following mapping:

  p1  = - sqrt(b) * a
  p2 = sqrt(b)

  Setting the parameters respecting the mapping, the two functions and therefore should give the same result.
  We then checked that the FFNNWaveFunction works as expected with two tests:

  1. Computing the variational energy with the two wave function must return the same result

  2. Computing the variational derivatives must result into the same results, asides for some costants.
  In particular:

  d(nn)/da = d(p1)/da * d(nn)/dp1 = -sqrt(b) * d(nn)/dp1

  d(nn)/db = d(p1)/db * d(nn)/dp1 + d(p2)/db * d(nn)/dp2 = ( -a / (2 * sqrt(b)) ) * d(nn)/dp1 + ( 1 / (2 * sqrt(b)) ) * d(nn)/dp2


*/





/*
  Hamiltonian describing a 1-particle harmonic oscillator:
  H  =  p^2 / 2m  +  1/2 * w^2 * x^2
*/
class HarmonicOscillator1D1P: public vmc::Hamiltonian
{

protected:
    double _w;

    mci::ObservableFunctionInterface * _clone() const final
    {
        return new HarmonicOscillator1D1P(_w);
    }

public:
    explicit HarmonicOscillator1D1P(double w):
            vmc::Hamiltonian(1 /*num space dimensions*/, 1 /*num particles*/) { _w = w; }

    // potential energy
    double localPotentialEnergy(const double * r) final
    {
        return (0.5*_w*_w*(*r)*(*r));
    }
};


/*
  Trial Wave Function for 1 particle in 1 dimension, that uses two variational parameters: a and b.
  Psi  =  exp( -b * (x-a)^2 )
  Notice that the corresponding probability density (sampling function) is Psi^2.
*/
class QuadrExponential1D1POrbital: public vmc::WaveFunction
{
protected:
    double _a, _b;

    mci::SamplingFunctionInterface * _clone() const final
    {
        return new QuadrExponential1D1POrbital(_a, _b, this->hasVD1());
    }

public:
    QuadrExponential1D1POrbital(double a, double b, bool flag_vd1 = false):
            vmc::WaveFunction(1 /*num space dimensions*/, 1 /*num particles*/, 1 /*num wf components*/, 2 /*num variational parameters*/, flag_vd1 /*VD1*/, false /*D1VD1*/, false /*D2VD1*/)
    {
        _a = a;
        _b = b;
    }

    void setVP(const double * in) final
    {
        _a = in[0];
        _b = in[1];
    }

    void getVP(double * out) const final
    {
        out[0] = _a;
        out[1] = _b;
    }

    void protoFunction(const double * x, double * out) final
    {
        /*
          Compute the sampling function proto value, used in acceptanceFunction()
        */
        *out = -2.*(_b*(x[0] - _a)*(x[0] - _a));
    }

    double acceptanceFunction(const double * protoold, const double * protonew) const final
    {
        /*
          Compute the acceptance probability
        */
        return exp(protonew[0] - protoold[0]);
    }

    void computeAllDerivatives(const double * x) final
    {
        _setD1DivByWF(0, -2.*_b*(x[0] - _a));
        _setD2DivByWF(0, -2.*_b + (-2.*_b*(x[0] - _a))*(-2.*_b*(x[0] - _a)));
        if (hasVD1()) {
            _setVD1DivByWF(0, 2.*_b*(x[0] - _a));
            _setVD1DivByWF(1, -(x[0] - _a)*(x[0] - _a));
        }
    }

    double computeWFValue(const double * protovalues) const final
    {
        return exp(0.5*protovalues[0]);
    }
};


int main()
{
    using namespace std;

    MPIVMC::Init(); // to not throw error when libraries are MPI-compiled

    // parameters
    const long Nmc = 20000l;
    const double TINY = 0.00001;

    // variational parameters
    const double a = 0.37;
    const double sqrtb = 1.18;
    const double b = sqrtb*sqrtb;
    const double p1 = -sqrtb*a;
    const double p2 = sqrtb;



    // create the QNets PolyNet with the right structure and parameters
    FeedForwardNeuralNetwork ffnn(2, 2, 2);
    ffnn.getNNLayer(0)->setActivationFunction(new GaussianActivationFunction());
    ffnn.getNNLayer(1)->setActivationFunction(new IdentityActivationFunction());
    ffnn.connectFFNN();
    ffnn.assignVariationalParameters(); // make all betas variational

    ffnn.setVariationalParameter(0, p1);
    ffnn.setVariationalParameter(1, p2);
    ffnn.setVariationalParameter(2, 0.);
    ffnn.setVariationalParameter(3, 1.);

    // printFFNNStructure(ffnn);

    // wrap ffnn by sannifa wrapper (to match interfaces)
    // and enable required derivatives
    QPolyWrapper wrapper(ffnn);
    wrapper.enableFirstDerivative();
    wrapper.enableSecondDerivative();
    wrapper.enableVariationalFirstDerivative();

    // NN wave function
    const int n = 1;
    ANNWaveFunction<QPolyWrapper> psi(n, n, wrapper);
    assert(psi.hasVD1());
    assert(!psi.hasD1VD1());
    assert(!psi.hasD2VD1());

    // gaussian wave function
    QuadrExponential1D1POrbital phi(a, b, true);

    // Hamiltonians
    HarmonicOscillator1D1P ham(1.);



    // --- Check that the energies are the same
    using namespace vmc;
    VMC vmc(psi, ham);
    vmc.getMCI().setSeed(1337);

    double energy[4];
    double d_energy[4];
    vmc.computeEnergy(Nmc, energy, d_energy);
    cout << "       Total Energy        = " << energy[0] << " +- " << d_energy[0] << endl;
    cout << "       Potential Energy    = " << energy[1] << " +- " << d_energy[1] << endl;
    cout << "       Kinetic (PB) Energy = " << energy[2] << " +- " << d_energy[2] << endl;
    cout << "       Kinetic (JF) Energy = " << energy[3] << " +- " << d_energy[3] << endl << endl;


    VMC vmc_check(phi, ham);
    vmc_check.getMCI().setSeed(1337);

    double energy_check[4];
    double d_energy_check[4];
    vmc_check.computeEnergy(Nmc, energy_check, d_energy_check);
    cout << "       Total Energy        = " << energy_check[0] << " +- " << d_energy_check[0] << endl;
    cout << "       Potential Energy    = " << energy_check[1] << " +- " << d_energy_check[1] << endl;
    cout << "       Kinetic (PB) Energy = " << energy_check[2] << " +- " << d_energy_check[2] << endl;
    cout << "       Kinetic (JF) Energy = " << energy_check[3] << " +- " << d_energy_check[3] << endl << endl;

    for (int i = 0; i < 4; ++i) {
        assert(abs(energy[i] - energy_check[i]) < 3.*sqrt(d_energy[i]*d_energy[i] + d_energy_check[i]*d_energy_check[i]));
    }


    // --- Check the variational derivatives
    const double dx = 0.2;
    double x = -1.;
    for (int i = 0; i < 10; ++i) {
        x = x + dx;

        phi.computeAllDerivatives(&x);
        psi.computeAllDerivatives(&x);

        const double dda_phi = phi.getVD1DivByWF(0);
        const double ddb_phi = phi.getVD1DivByWF(1);


        const double ddp1_psi = psi.getVD1DivByWF(0);
        const double ddp2_psi = psi.getVD1DivByWF(1);


        // cout << dda_phi << " == " << - ddp1_psi * sqrtb << " ? " << endl;
        assert(abs(dda_phi - (-ddp1_psi*sqrtb)) < TINY);

        // cout << ddb_phi << " == " << ddp2_psi / (2. * sqrtb) - ddp1_psi * a / (2. * sqrtb) << " ? " << endl;
        assert(abs(ddb_phi - (ddp2_psi/(2.*sqrtb) - ddp1_psi*a/(2.*sqrtb))) < TINY);
    }

    MPIVMC::Finalize();

    return 0;
}
