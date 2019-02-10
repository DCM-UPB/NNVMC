#include "nnvmc/SannifaWaveFunction.hpp"

#include <cmath>

// -- Constructor and destructor

SannifaWaveFunction::SannifaWaveFunction(const int &nspacedim, const int &npart, ANNFunctionInterface * ann)
    :WaveFunction(nspacedim, npart, 1, ann->getNVariationalParameters(),
                  ann->hasVariationalFirstDerivative(), ann->hasCrossFirstDerivative(), ann->hasCrossSecondDerivative()),
     _ann(ann)
{
    if (ann->getNInput() != nspacedim*npart) {
        throw std::invalid_argument( "ANN number of inputs does not fit the nspacedime and npart" );
    }
    if (ann->getNOutput() != 1) {
        throw std::invalid_argument( "ANN number of output does not fit the wave function requirement (only one value)" );
    }
    if (!ann->hasFirstDerivative()) {
        throw std::invalid_argument( "ANN does not provide at least the first derivative to compute energies.");
    }
}

// --- interface for manipulating the variational parameters
void SannifaWaveFunction::setVP(const double *vp){
    _ann->setVariationalParameters(vp);
}

void SannifaWaveFunction::getVP(double *vp){
    _ann->getVariationalParameters(vp);
}

// --- methods inherited from MCISamplingFunctionInterface

void SannifaWaveFunction::samplingFunction(const double * in, double * out){
    _ann->evaluate(in, false); // false -> no gradient
    out[0] = pow(_ann->getOutput(0), 2);
}

double SannifaWaveFunction::getAcceptance(const double * protoold, const double * protonew){
    if ((protoold[0] == 0.) && (protonew[0] != 0.)){
        return 1.;
    } else if ((protoold[0] == 0.) && (protonew[0] == 0.)) {
        return 0.;
    }

    return protonew[0]/protoold[0];
}

// --- computation of the derivatives

void SannifaWaveFunction::computeAllDerivatives(const double *in){
    _ann->evaluate(in, true); // true -> with gradient

    const double wf_value = _ann->getOutput(0);
    for (int id1=0; id1<getTotalNDim(); ++id1){
        _setD1DivByWF(id1, _ann->getFirstDerivative(0, id1) / wf_value);
    }

    for (int id2=0; id2<getTotalNDim(); ++id2){
        _setD2DivByWF(id2, _ann->getSecondDerivative(0, id2) / wf_value);
    }

    if (hasVD1()){
        for (int ivd1=0; ivd1<getNVP(); ++ivd1){
            _setVD1DivByWF(ivd1, _ann->getVariationalFirstDerivative(0, ivd1) / wf_value);
        }
    }

    if (hasD1VD1()){
        for (int id1=0; id1<getTotalNDim(); ++id1){
            for (int ivd1=0; ivd1<getNVP(); ++ivd1){
                _setD1VD1DivByWF(id1, ivd1, _ann->getCrossFirstDerivative(0, id1, ivd1) / wf_value);
            }
        }
    }

    if (hasD2VD1()){
        for (int id2=0; id2<getTotalNDim(); ++id2){
            for (int ivd2=0; ivd2<getNVP(); ++ivd2){
                _setD2VD1DivByWF(id2, ivd2, _ann->getCrossSecondDerivative(0, id2, ivd2) / wf_value);
            }
        }
    }
}

double SannifaWaveFunction::computeWFValue(const double * protovalues)
{
    return sqrt(protovalues[0]);
}
