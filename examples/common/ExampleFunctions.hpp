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


#endif
