#ifndef SANNIFA_WAVE_FUNCTION
#define SANNIFA_WAVE_FUNCTION

#include "vmc/WaveFunction.hpp"
#include "sannifa/ANNFunctionInterface.hpp"

#include <stdexcept>


class SannifaWaveFunction: public WaveFunction{

private:
    ANNFunctionInterface * _ann; // we will point to externally owned ANN

public:
    // --- Constructor and destructor
    // IMPORTANT: The provided ann should be ready to use
    // and have all required derivatives enabled
    SannifaWaveFunction(const int &nspacedim, const int &npart, ANNFunctionInterface * ann);
    ~SannifaWaveFunction(){};


    // --- Getters
    ANNFunctionInterface * getANN(){return _ann;}

    // --- interface for manipulating the variational parameters
    void setVP(const double *vp);
    void getVP(double *vp);

    // --- methods inherited from MCISamplingFunctionInterface
    // wave function values that will be used to compute the acceptance
    void samplingFunction(const double * in, double * out);
    // MCI acceptance starting from the new and old sampling functions
    double getAcceptance(const double * protoold, const double * protonew);

    // --- computation of the derivatives / wf value
    void computeAllDerivatives(const double *in);
    double computeWFValue(const double * protovalues);
};


#endif
