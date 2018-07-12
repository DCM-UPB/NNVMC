#include "MCIntegrator.hpp"
#include "MCIObservableFunctionInterface.hpp"
#include "FeedForwardNeuralNetwork.hpp"

#include <iostream>

/*
  Build an example for MCI++:

  compute simultaneously the integrals

  \int_{-10}^{+10} dx nn(x) nn(x)^2
  \int_{-10}^{+10} dx (d/dx nn(x)) nn(x)^2
  \int_{-10}^{+10} dx (d^2/dx^2 nn(x)) nn(x)^2
  \int_{-10}^{+10} dx (d/dbeta_i nn(x)) nn(x)^2    --- beta are the variational parameters

  (nn(x) is a normalized neural network) with MC and a non-MC method.

  This is done by using two NNs: one without any derivative substrate, and one
  with the first and second derivative substrate.
  The fast NN without derivatives is used for sampling, the one with the substrates
  instead will be used for computing all the derivatives, by making use of a
  call-back function.

  The use of a call-back example is not necessary here, it could have been without.
  It is done in this way just for illustrative purposes.

*/





class MyInterfaces: public MCISamplingFunctionInterface, public MCIObservableFunctionInterface, public MCICallBackOnAcceptanceInterface{
private:
    FeedForwardNeuralNetwork * _bare_ffnn;
    FeedForwardNeuralNetwork * _deriv_ffnn;

public:
    MyInterfaces(FeedForwardNeuralNetwork * bare_ffnn, FeedForwardNeuralNetwork * deriv_ffnn)
        : MCISamplingFunctionInterface(1, 1),
            MCIObservableFunctionInterface(1, 3+deriv_ffnn->getNVariationalParameters()),
            MCICallBackOnAcceptanceInterface(1){
        _bare_ffnn = bare_ffnn;
        _deriv_ffnn = deriv_ffnn;
    }

    void samplingFunction(const double *in, double * protovalues){
        _bare_ffnn->setInput(in);
        _bare_ffnn->FFPropagate();
        const double v = _bare_ffnn->getOutput(0);
        protovalues[0] = v*v;
    }

    double getAcceptance(const double * protoold, const double * protonew){
        if (protoold[0] == 0.) return 1.;
        return protonew[0]/protoold[0];
    }

    void callBackFunction(const double * in, const bool flag_observation){
        if (flag_observation){
            _deriv_ffnn->setInput(in);
            _deriv_ffnn->FFPropagate();
        }
    }

    void observableFunction(const double * in, double *out){
        out[0] = _deriv_ffnn->getOutput(0);
        out[1] = _deriv_ffnn->getFirstDerivative(0, 0);
        out[2] = _deriv_ffnn->getSecondDerivative(0, 0);
        for (int i=0; i<_deriv_ffnn->getNBeta(); ++i){
            out[3+i] = _deriv_ffnn->getVariationalFirstDerivative(0, i);
        }
    }

};






int main(){
    using namespace std;

    const long NMC = 400000;

    FeedForwardNeuralNetwork * deriv_ffnn = new FeedForwardNeuralNetwork(2, 10, 2);
    deriv_ffnn->connectFFNN();
    deriv_ffnn->assignVariationalParameters();
    FeedForwardNeuralNetwork * bare_ffnn = new FeedForwardNeuralNetwork(deriv_ffnn);
    deriv_ffnn->addFirstDerivativeSubstrate();
    deriv_ffnn->addSecondDerivativeSubstrate();
    deriv_ffnn->addVariationalFirstDerivativeSubstrate();

    MyInterfaces * my_interfaces = new MyInterfaces(bare_ffnn, deriv_ffnn);

    MCI * mci = new MCI(1);

    double ** irange = new double*[1];
    irange[0] = new double[2];
    irange[0][0] = -10.;
    irange[0][1] = +10.;

    mci->setIRange(irange);

    mci->addObservable(my_interfaces);
    mci->addSamplingFunction(my_interfaces);
    mci->addCallBackOnAcceptance(my_interfaces);
    double * avg = new double[3+deriv_ffnn->getNBeta()];
    double * err = new double[3+deriv_ffnn->getNBeta()];
    mci->integrate(NMC, avg, err);

    cout << endl;
    cout << "int_{-10}^{+10} dx nn(x) nn(x)^2  =  " << avg[0] << " +- " << err[0] << endl;
    cout << "int_{-10}^{+10} dx (d/dx nn(x)) nn(x)^2  =  " << avg[1] << " +- " << err[1] << endl;
    cout << "int_{-10}^{+10} dx (d^2/dx^2 nn(x)) nn(x)^2  =  " << avg[2] << " +- " << err[2] << endl;
    for (int i=0; i<deriv_ffnn->getNBeta(); ++i){
        cout << "int_{-10}^{+10} dx (d/dbeta_" << i << " nn(x)) nn(x)^2  =  " << avg[3+i] << " +- " << err[3+i] << endl;
    }
    cout << endl;



    delete avg;
    delete err;
    delete[] irange[0];
    delete[] irange;
    delete mci;
    delete my_interfaces;
    delete bare_ffnn;
    delete deriv_ffnn;

    return 0;
}
