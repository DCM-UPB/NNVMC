#include "MCIntegrator.hpp"
#include "MCIObservableFunctionInterface.hpp"
#include "FeedForwardNeuralNetwork.hpp"

#include <iostream>

/*
  Build an example for MCI++:

  compute the integral

  \int_{-10}^{+10} dx x^2 nn(x)^2

  (nn(x) is a normalized neural network) with MC and a non-MC method.

  Do the same with

  \int_{-10}^{+10} dx (- \nabla^2 log(nn(x))) nn(x)^2

  (nabla^2 is the laplacian operator). This is a kinetic energy.
  In the following code by nabla2 we will mean: (- \nabla^2 log(nn(x)))

*/





class NN2: public MCIObservableFunctionInterface{
private:
    FeedForwardNeuralNetwork * _ffnn;

public:
    NN2(FeedForwardNeuralNetwork * ffnn): MCIObservableFunctionInterface(1, 1){
        _ffnn = ffnn;
    }

    virtual void observableFunction(const double * in, double *out){
        _ffnn->setInput(in);
        _ffnn->FFPropagate();
        const double v = _ffnn->getOutput(0);
        out[0] = v*v;
    }

};



class NN2X2: public MCIObservableFunctionInterface{
private:
    FeedForwardNeuralNetwork * _ffnn;

public:
    NN2X2(FeedForwardNeuralNetwork * ffnn): MCIObservableFunctionInterface(1, 1){
        _ffnn = ffnn;
    }

    virtual void observableFunction(const double * in, double *out){
        _ffnn->setInput(in);
        _ffnn->FFPropagate();
        const double v = _ffnn->getOutput(0);
        out[0] = v*v*in[0]*in[0];
    }

};


class X2: public MCIObservableFunctionInterface{
private:
    FeedForwardNeuralNetwork * _ffnn;

public:
    X2(): MCIObservableFunctionInterface(1, 1){}

    virtual void observableFunction(const double * in, double *out){
        out[0] = in[0]*in[0];
    }

};




class NN2Sampling: public MCISamplingFunctionInterface{
private:
    FeedForwardNeuralNetwork * _ffnn;
public:
    NN2Sampling(FeedForwardNeuralNetwork * ffnn): MCISamplingFunctionInterface(1, 1){
        _ffnn = ffnn;
    }

    void samplingFunction(const double *in, double * protovalues){
        _ffnn->setInput(in);
        _ffnn->FFPropagate();
        const double v = _ffnn->getOutput(0);
        protovalues[0] = v*v;
    }

    double getAcceptance(const double * protoold, const double * protonew){
        return protonew[0]/protoold[0];
    }
};


class NN2Nabla2: public MCIObservableFunctionInterface{
private:
    FeedForwardNeuralNetwork * _ffnn;

public:
    NN2Nabla2(FeedForwardNeuralNetwork * ffnn): MCIObservableFunctionInterface(1, 1){
        _ffnn = ffnn;
    }

    virtual void observableFunction(const double * in, double *out){
        _ffnn->setInput(in);
        _ffnn->FFPropagate();
        const double v = _ffnn->getOutput(0);
        const double d2 = _ffnn->getSecondDerivative(0, 0);
        out[0] = -v*d2;
    }

};


class Nabla2: public MCIObservableFunctionInterface{
private:
    FeedForwardNeuralNetwork * _ffnn;

public:
    Nabla2(FeedForwardNeuralNetwork * ffnn): MCIObservableFunctionInterface(1, 1){
        _ffnn = ffnn;
    }

    virtual void observableFunction(const double * in, double *out){
        _ffnn->setInput(in);
        _ffnn->FFPropagate();
        const double v = _ffnn->getOutput(0);
        const double d2 = _ffnn->getSecondDerivative(0, 0);
        out[0] = -d2/v;
    }

};




int main(){
    using namespace std;

    const long NMC = 400000;
    const double DX = 0.0001;

    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(2, 10, 2);
    ffnn->connectFFNN();
    ffnn->addFirstDerivativeSubstrate();
    ffnn->addSecondDerivativeSubstrate();

    MCI * mci = new MCI(1);

    double ** irange = new double*[1];
    irange[0] = new double[2];
    irange[0][0] = -10.;
    irange[0][1] = +10.;

    mci->setIRange(irange);


    // compute   \int_{-10}^{+10} dx nn(x)^2    which is the normalization factor
    NN2 * nn2 = new NN2(ffnn);
    mci->addObservable(nn2);
    double * nn2_av = new double;
    double * nn2_er = new double;
    mci->integrate(NMC, nn2_av, nn2_er);
    cout << "Compute the normalization factor:";
    cout << "    normalization = " << *nn2_av << " +- " << *nn2_er << endl;





    // ---   \int_{-10}^{+10} dx x^2 nn(x)^2

    cout << endl << endl << "We now compute the integral" << endl;
    cout << "    int_{-10}^{+10} dx x^2 nn(x)^2" << endl << endl;


    // compute   \int_{-10}^{+10} dx x^2 nn(x)^2    MC without sampling
    NN2X2 * nn2x2 = new NN2X2(ffnn);
    mci->clearObservables();
    mci->addObservable(nn2x2);
    double * nn2x2_av = new double;
    double * nn2x2_er = new double;
    mci->integrate(NMC, nn2x2_av, nn2x2_er);
    mci->clearObservables();
    cout << "1. MC without sampling: ";
    cout << *nn2x2_av / *nn2_av << " +- " << *nn2x2_er / *nn2_av << endl << endl;


    // compute   \int_{-10}^{+10} dx x^2 nn(x)^2    MC with sampling
    X2 * x2 = new X2();
    mci->addObservable(x2);
    NN2Sampling * nn2_samp = new NN2Sampling(ffnn);
    mci->addSamplingFunction(nn2_samp);
    double * nn2x2_samp_av = new double;
    double * nn2x2_samp_er = new double;
    mci->integrate(NMC, nn2x2_samp_av, nn2x2_samp_er);
    mci->clearObservables();
    mci->clearSamplingFunctions();
    cout << "2. MC with sampling: ";
    cout << *nn2x2_samp_av << " +- " << *nn2x2_samp_er << endl << endl;


    // compute   \int_{-10}^{+10} dx x^2 nn(x)^2    direct integral
    double x = irange[0][0];
    double y = 0.;
    double integral = 0.;
    while (x < irange[0][1]){
        nn2x2->observableFunction(&x, &y);
        integral += y*DX;
        x += DX;
    }
    cout << "3. Direct integral: ";
    cout << integral / *nn2_av << endl << endl;





    // ---   \int_{-10}^{+10} dx (- \nabla^2 log(nn(x))) nn(x)^2

    cout << endl << endl << "We now compute the integral" << endl;
    cout << "    int_{-10}^{+10} dx (- nabla^2 log(nn(x))) nn(x)^2" << endl << endl;


    // compute   \int_{-10}^{+10} dx (- \nabla^2 log(nn(x))) nn(x)^2   MC without sampling
    NN2Nabla2 * nn2nabla2 = new NN2Nabla2(ffnn);
    mci->addObservable(nn2nabla2);
    double * nn2nabla2_av = new double;
    double * nn2nabla2_er = new double;
    mci->integrate(NMC, nn2nabla2_av, nn2nabla2_er);
    mci->clearObservables();
    cout << "1. MC without sampling: ";
    cout << *nn2nabla2_av / *nn2_av << " +- " << *nn2nabla2_er / *nn2_av << endl << endl;

    // compute   \int_{-10}^{+10} dx (- \nabla^2 log(nn(x))) nn(x)^2    MC with sampling
    Nabla2 * nabla2 = new Nabla2(ffnn);
    mci->addObservable(nabla2);
    mci->addSamplingFunction(nn2_samp);
    double * nn2nabla2_samp_av = new double;
    double * nn2nabla2_samp_er = new double;
    mci->integrate(NMC, nn2nabla2_samp_av, nn2nabla2_samp_er);
    mci->clearObservables();
    mci->clearSamplingFunctions();
    cout << "2. MC with sampling: ";
    cout << *nn2nabla2_samp_av << " +- " << *nn2nabla2_samp_er << endl << endl;

    // compute   \int_{-10}^{+10} dx (- \nabla^2 log(nn(x))) nn(x)^2    direct integral
    x = irange[0][0];
    y = 0.;
    integral = 0.;
    while (x < irange[0][1]){
        nn2nabla2->observableFunction(&x, &y);
        integral += y*DX;
        x += DX;
    }
    cout << "3. Direct integral: ";
    cout << integral / *nn2_av << endl;






    delete nn2nabla2_samp_er;
    delete nn2nabla2_samp_av;
    delete nabla2;
    delete nn2nabla2_er;
    delete nn2nabla2_av;
    delete nn2nabla2;

    delete nn2x2_er;
    delete nn2x2_av;
    delete nn2_er;
    delete nn2_av;
    delete nn2;
    delete[] irange[0];
    delete[] irange;
    delete mci;
    delete ffnn;

    return 0;
}
