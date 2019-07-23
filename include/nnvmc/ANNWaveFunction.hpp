#ifndef NNVMC_ANNWAVEFUNCTION_HPP
#define NNVMC_ANNWAVEFUNCTION_HPP

#include "vmc/WaveFunction.hpp"
#include "sannifa/Sannifa.hpp"


#include <stdexcept>

// ANNWaveFunction implements a vmc::WaveFunction based on a
// ANN function of type ANNType. ANNType is expected to derive
// from or behave like a Sannifa wrapper (see sannifa library).
// -> You are not required to derive from Sannifa, but doing so
// guarantees compatibility with this template.
template <class ANNType>
class ANNWaveFunction final: public vmc::WaveFunction
{

private:
    ANNType * const _ann; // a single ann is copy-created at construction and deleted in destructor

    mci::SamplingFunctionInterface * _clone() const final
    {
        return new ANNWaveFunction(this->getNSpaceDim(), this->getNPart(), *_ann);
    }

public:

    // --- Constructor and destructor
    // IMPORTANT: The provided ann should be ready to use
    // and have all required derivatives enabled
    ANNWaveFunction(int nspacedim, int npart, const ANNType &ann):
            vmc::WaveFunction(nspacedim, npart, 1, ann.getNVariationalParameters(), ann.hasVariationalFirstDerivative(), ann.hasCrossFirstDerivative(), ann.hasCrossSecondDerivative()),
            _ann(new ANNType(ann))
    {
        if (_ann->getNInput() != nspacedim*npart) {
            throw std::invalid_argument("ANN number of inputs does not fit the nspacedime and npart");
        }
        if (_ann->getNOutput() != 1) {
            throw std::invalid_argument("ANN number of output does not fit the wave function requirement (only one value)");
        }
        if (!_ann->hasFirstDerivative()) {
            throw std::invalid_argument("ANN does not provide at least the first derivative to compute energies.");
        }
    }

    ~ANNWaveFunction() final { delete _ann; }

    // --- const ann access
    const ANNType &getANN() { return *_ann; }

    // --- interface for manipulating the variational parameters
    void setVP(const double vp[]) final { _ann->setVariationalParameters(vp); }
    void getVP(double vp[]) const final { _ann->getVariationalParameters(vp); }

    // --- methods inherited from MCISamplingFunctionInterface
    // wave function values that will be used to compute the acceptance
    void protoFunction(const double in[], double protov[]) final
    {
        _ann->evaluate(in, false); // false -> no gradient
        protov[0] = _ann->getOutput(0);
    }

    // MCI acceptance starting from the new and old sampling functions
    double acceptanceFunction(const double protoold[], const double protonew[]) const final
    {
        if (protoold[0] == 0.) {
            return (protonew[0] != 0.) ? 1. : 0.;
        }
        return (protonew[0]*protonew[0])/(protoold[0]*protoold[0]);
    }

    // --- computation of the derivatives / wf value
    void computeAllDerivatives(const double in[]) final
    {
        _ann->evaluate(in, true); // true -> with gradient
        const double wf_value = _ann->getOutput(0);

        for (int id1 = 0; id1 < getTotalNDim(); ++id1) {
            _setD1DivByWF(id1, _ann->getFirstDerivative(0, id1)/wf_value);
        }

        if (_ann->hasSecondDerivative()) {
            for (int id2 = 0; id2 < getTotalNDim(); ++id2) {
                _setD2DivByWF(id2, _ann->getSecondDerivative(0, id2)/wf_value);
            }
        }

        if (hasVD1()) {
            for (int ivd1 = 0; ivd1 < getNVP(); ++ivd1) {
                _setVD1DivByWF(ivd1, _ann->getVariationalFirstDerivative(0, ivd1)/wf_value);
            }
        }

        if (hasD1VD1()) {
            for (int id1 = 0; id1 < getTotalNDim(); ++id1) {
                for (int ivd1 = 0; ivd1 < getNVP(); ++ivd1) {
                    _setD1VD1DivByWF(id1, ivd1, _ann->getCrossFirstDerivative(0, id1, ivd1)/wf_value);
                }
            }
        }

        if (hasD2VD1()) {
            for (int id2 = 0; id2 < getTotalNDim(); ++id2) {
                for (int ivd2 = 0; ivd2 < getNVP(); ++ivd2) {
                    _setD2VD1DivByWF(id2, ivd2, _ann->getCrossSecondDerivative(0, id2, ivd2)/wf_value);
                }
            }
        }
    }

    double computeWFValue(const double protovalues[]) const final
    {
        return protovalues[0];
    }
};


#endif
