#ifndef FFNN_WAVE_FUNCTION
#define FFNN_WAVE_FUNCTION

#include "vmc/WaveFunction.hpp"
#include "ffnn/FeedForwardNeuralNetwork.hpp"

#include <stdexcept>



class FFNNWaveFunction: public WaveFunction{

private:
    FeedForwardNeuralNetwork * _bare_ffnn;   // FFNN without derivatives, used for sampling
    FeedForwardNeuralNetwork * _deriv_ffnn;   // FFNN with derivatives, used for computing all the derivatives

public:
    // --- Constructor and destructor
    // IMPORTANT: The provided ffnn should be ready to use (connected), but should not contain any substrate,
    // as they will inside this class depending on the needs
    FFNNWaveFunction(const int &nspacedim, const int &npart, FeedForwardNeuralNetwork * ffnn, bool flag_vd1=false, bool flag_d1vd1=false, bool flag_d2vd1=false);
    ~FFNNWaveFunction();



    // --- Getters
    FeedForwardNeuralNetwork * getBareFFNN(){return _bare_ffnn;}
    FeedForwardNeuralNetwork * getDerivFFNN(){return _deriv_ffnn;}


    // --- interface for manipulating the variational parameters
    void setVP(const double *vp);
    void getVP(double *vp);

    // --- methods herited from MCISamplingFunctionInterface
    // wave function values that will be used to compute the acceptance
    void samplingFunction(const double * in, double * out);
    // MCI acceptance starting from the new and old sampling functions
    double getAcceptance(const double * protoold, const double * protonew);

    // --- computation of the derivatives / wf value
    void computeAllDerivatives(const double *in);
    double computeWFValue(const double * protovalues);
};


#endif
