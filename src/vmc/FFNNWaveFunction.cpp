#include "FFNNWaveFunction.hpp"

#include <cmath>



// --- interface for manipulating the variational parameters
void FFNNWaveFunction::setVP(const double *vp){
    for (int i=0; i<_bare_ffnn->getNVariationalParameters(); ++i){
        _bare_ffnn->setVariationalParameter(i, vp[i]);
        _deriv_ffnn->setVariationalParameter(i, vp[i]);
    }
}


void FFNNWaveFunction::getVP(double *vp){
    for (int i=0; i<_bare_ffnn->getNVariationalParameters(); ++i){
        vp[i] = _bare_ffnn->getVariationalParameter(i);
    }
}




// --- methods herited from MCISamplingFunctionInterface

void FFNNWaveFunction::samplingFunction(const double * in, double * out){
    _bare_ffnn->setInput(in);
    _bare_ffnn->FFPropagate();
    out[0] = pow(_bare_ffnn->getOutput(0), 2);
}


double FFNNWaveFunction::getAcceptance(const double * protoold, const double * protonew){
    if ((protoold[0] == 0.) && (protonew[0] != 0.)){
        return 1.;
    } else if ((protoold[0] == 0.) && (protonew[0] == 0.)) {
        return 0.;
    }

    return protonew[0]/protoold[0];
}




// --- computation of the derivatives

void FFNNWaveFunction::computeAllDerivatives(const double *in){
    _deriv_ffnn->setInput(in);
    _deriv_ffnn->FFPropagate();

    const double wf_value = _deriv_ffnn->getOutput(0);

    for (int id1=0; id1<getTotalNDim(); ++id1){
        _setD1DivByWF(id1, _deriv_ffnn->getFirstDerivative(0, id1) / wf_value);
    }

    for (int id2=0; id2<getTotalNDim(); ++id2){
        _setD2DivByWF(id2, _deriv_ffnn->getSecondDerivative(0, id2) / wf_value);
    }

    if (hasVD1()){
        for (int ivd1=0; ivd1<getNVP(); ++ivd1){
            _setVD1DivByWF(ivd1, _deriv_ffnn->getVariationalFirstDerivative(0, ivd1) / wf_value);
        }
    }

    if (hasD1VD1()){
        for (int id1=0; id1<getTotalNDim(); ++id1){
            for (int ivd1=0; ivd1<getNVP(); ++ivd1){
                _setD1VD1DivByWF(id1, ivd1, _deriv_ffnn->getCrossFirstDerivative(0, id1, ivd1) / wf_value);
            }
        }
    }

    if (hasD2VD1()){
        for (int id2=0; id2<getTotalNDim(); ++id2){
            for (int ivd1=0; ivd1<getNVP(); ++ivd1){
                _setD1VD1DivByWF(id2, ivd1, _deriv_ffnn->getCrossSecondDerivative(0, id2, ivd1) / wf_value);
            }
        }
    }

}




// -- Constructor and destructor


FFNNWaveFunction::FFNNWaveFunction(const int &nspacedim, const int &npart, FeedForwardNeuralNetwork * ffnn, bool flag_vd1, bool flag_d1vd1, bool flag_d2vd1)
    :WaveFunction(nspacedim, npart, 1, ffnn->getNVariationalParameters(), flag_vd1, flag_d1vd1, flag_d2vd1){
    if (ffnn->getNInput() != nspacedim*npart)
        throw std::invalid_argument( "FFNN number of inputs does not fit the nspacedime and npart" );

    if (ffnn->getNOutput() != 1)
        throw std::invalid_argument( "FFNN number of output does not fit the wave function requirement (only one value)" );

    if (ffnn->hasFirstDerivativeSubstrate() || ffnn->hasSecondDerivativeSubstrate() || ffnn->hasVariationalFirstDerivativeSubstrate() ||
    ffnn->hasCrossFirstDerivativeSubstrate() || ffnn->hasCrossSecondDerivativeSubstrate())
        throw std::invalid_argument( "FFNN should not have any substrate" );

    _bare_ffnn = new FeedForwardNeuralNetwork(ffnn);
    _deriv_ffnn = new FeedForwardNeuralNetwork(ffnn);
    _deriv_ffnn->addFirstDerivativeSubstrate();
    _deriv_ffnn->addSecondDerivativeSubstrate();
    if (flag_vd1) _deriv_ffnn->addVariationalFirstDerivativeSubstrate();
    if (flag_d1vd1) _deriv_ffnn->addCrossFirstDerivativeSubstrate();
    if (flag_d2vd1) _deriv_ffnn->addCrossSecondDerivativeSubstrate();
}



FFNNWaveFunction::~FFNNWaveFunction(){
    delete _bare_ffnn;
    _bare_ffnn = 0;
    delete _deriv_ffnn;
    _deriv_ffnn = 0;
}
