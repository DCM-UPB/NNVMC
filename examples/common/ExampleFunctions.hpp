#ifndef NNVMC_EXAMPLEFUNCTIONS_HPP
#define NNVMC_EXAMPLEFUNCTIONS_HPP

#include <cmath>
#include <algorithm>

#include "vmc/Hamiltonian.hpp"
#include "vmc/WaveFunction.hpp"

#include "nfm/NoisyFunction.hpp"

#include "sannifa/Sannifa.hpp"

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
    double localPotentialEnergy(const double r[]) final
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

    void setVP(const double in[]) final
    {
        _a = in[0];
        _b = in[1];
    }

    void getVP(double out[]) const final
    {
        out[0] = _a;
        out[1] = _b;
    }

    void protoFunction(const double x[], double out[]) final
    {
        /*
          Compute the sampling function proto value, used in acceptanceFunction()
        */
        *out = -2.*(_b*(x[0] - _a)*(x[0] - _a));
    }

    double acceptanceFunction(const double protoold[], const double protonew[]) const final
    {
        /*
          Compute the acceptance probability
        */
        return exp(protonew[0] - protoold[0]);
    }

    void computeAllDerivatives(const double x[]) final
    {
        _setD1DivByWF(0, -2.*_b*(x[0] - _a));
        _setD2DivByWF(0, -2.*_b + (-2.*_b*(x[0] - _a))*(-2.*_b*(x[0] - _a)));
        if (hasVD1()) {
            _setVD1DivByWF(0, 2.*_b*(x[0] - _a));
            _setVD1DivByWF(1, -(x[0] - _a)*(x[0] - _a));
        }
    }

    double computeWFValue(const double protovalues[]) const final
    {
        return exp(0.5*protovalues[0]);
    }
};


/*
  Trial Wave Function for 1 particle in 1 dimension, that uses one variational parameters: b.
  Psi  =  exp( -b * x^2 )
  Notice that the corresponding probability density (sampling function) is Psi^2.
*/
class Gaussian1D1POrbital: public vmc::WaveFunction
{
protected:
    double _b;

    mci::SamplingFunctionInterface * _clone() const final
    {
        return new Gaussian1D1POrbital(_b, _flag_vd1);
    }

public:
    explicit Gaussian1D1POrbital(double b, bool flag_vd1 = false):
            vmc::WaveFunction(1, 1, 1, 1, flag_vd1, false, false)
    {
        _b = b;
    }

    void setVP(const double in[]) final
    {
        _b = *in;
    }

    void getVP(double out[]) const final
    {
        *out = _b;
    }

    void protoFunction(const double in[], double out[]) final
    {
        *out = -2.*_b*(*in)*(*in);
    }

    double acceptanceFunction(const double protoold[], const double protonew[]) const final
    {
        return exp(protonew[0] - protoold[0]);
    }

    void computeAllDerivatives(const double in[]) final
    {
        _setD1DivByWF(0, -2.*_b*(*in));
        _setD2DivByWF(0, -2.*_b + 4.*_b*_b*(*in)*(*in));
        if (hasVD1()) {
            _setVD1DivByWF(0, (-(*in)*(*in)));
        }
    }

    double computeWFValue(const double protovalues[]) const final
    {
        return exp(0.5*protovalues[0]);
    }
};


// Simple Cost Function for NN fitting
//
// Basic all-data mean-squares cost without regularization.
// Suboptimal efficiency, because no combined fgrad is implemented.
class NNFitCost: public nfm::NoisyFunctionWithGradient
{
protected:
    Sannifa * const _ann;
    const int _ndata;
    const double * const _xdata;
    const double * const _ydata;
    const bool _useError;

public:
    explicit NNFitCost(Sannifa * ann, int ndata, const double xdata[] /*ndata*ninput*/, const double ydata[] /*ndata*/, bool useError = true):
            nfm::NoisyFunctionWithGradient(ann->getNVariationalParameters(), false/*no grad errors*/), _ann(ann), _ndata(ndata), _xdata(xdata), _ydata(ydata), _useError(useError) {}

    nfm::NoisyValue f(const std::vector<double> &in) override
    {
        _ann->setVariationalParameters(in.data());
        nfm::NoisyValue y{0., 0.};
        std::vector<double> resis(static_cast<size_t>(_ndata));
        for (int i = 0; i < _ndata; ++i) {
            _ann->evaluate(_xdata + i, false);
            resis[i] = pow(_ann->getOutput(0) - _ydata[i], 2);
            y.val += resis[i];
        }
        y.val /= _ndata;
        if (_useError) { // error of the mean estimation
            for (int i = 0; i < _ndata; ++i) {
                y.err += pow(resis[i] - y.val, 2);
            }
            y.err = sqrt(y.err/((_ndata - 1.)*_ndata));
        }
        return y;
    }

    void grad(const std::vector<double> &in, nfm::NoisyGradient &grad) override
    {
        _ann->setVariationalParameters(in.data());
        std::fill(grad.val.begin(), grad.val.end(), 0.);
        for (int i = 0; i < _ndata; ++i) {
            _ann->evaluate(_xdata + i, true);
            const double diff2 = 2.*(_ann->getOutput(0) - _ydata[i]) / _ndata;
            for (int j = 0; j < _ann->getNVariationalParameters(); ++j) {
                grad.val[j] -= diff2*_ann->getVariationalFirstDerivative(0, j);
            }
        }
    }
};


// Hydrogen Molecule

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

#endif
