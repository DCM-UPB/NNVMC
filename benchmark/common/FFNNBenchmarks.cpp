#include <iostream>
#include <cmath>
#include <tuple>

#include "FeedForwardNeuralNetwork.hpp"
#include "Timer.cpp"

double benchmark_FFPropagate(FeedForwardNeuralNetwork * const ffnn, const double * const * const xdata, const int neval) {
    Timer * const timer = new Timer();
    double time;

    timer->reset();
    for (int i=0; i<neval; ++i) {
        ffnn->setInput(xdata[i]);
        ffnn->FFPropagate();
    }
    time = timer->elapsed();

    delete timer;
    return time;
}

std::pair<double, double> sample_benchmark_FFPropagate(FeedForwardNeuralNetwork * ffnn, const double * const * const xdata, const int neval, const int nruns) {
    double times[nruns];
    double mean = 0., err = 0.;

    for (int i=0; i<nruns; ++i) {
        times[i] = benchmark_FFPropagate(ffnn, xdata, neval);
        mean += times[i];
    }
    mean /= nruns;
    for (int i=0; i<nruns; ++i) err += pow(times[i]-mean, 2);
    err /= (nruns-1)*nruns; // variance of the mean
    err = sqrt(err); // standard error of the mean

    const std::pair<double, double> result(mean, err);
    return result;
}

double benchmark_actf_derivs(ActivationFunctionInterface * actf, const double * const xdata, const int neval, const bool flag_d1 = true, const bool flag_d2 = true, const bool flag_d3 = true, const bool flag_fad = true) {
    Timer * const timer = new Timer();
    double time, v, v1d, v2d, v3d;

    if (flag_fad) {
        timer->reset();
        for (int i=0; i<neval; ++i) {
            actf->fad(xdata[i], v, v1d, v2d, v3d, flag_d1, flag_d2, flag_d3);
        }
        time = timer->elapsed();
    }
    else {
        timer->reset();
        for (int i=0; i<neval; ++i) {
            v = actf->f(xdata[i]);
            v1d = flag_d1 ? actf->f1d(xdata[i]) : 0.0;
            v2d = flag_d2 ? actf->f2d(xdata[i]) : 0.0;
            v3d = flag_d3 ? actf->f3d(xdata[i]) : 0.0;
        }
        time = timer->elapsed();
    }

    delete timer;
    return time;
}

std::pair<double, double> sample_benchmark_actf_derivs(ActivationFunctionInterface * actf, const double * const xdata, const int neval, const int nruns, const bool flag_d1 = true, const bool flag_d2 = true, const bool flag_d3 = true, const bool flag_fad = true) {
    double times[nruns];
    double mean = 0., err = 0.;

    for (int i=0; i<nruns; ++i) {
        times[i] = benchmark_actf_derivs(actf, xdata, neval, flag_d1, flag_d2, flag_d3, flag_fad);
        mean += times[i];
    }
    mean /= nruns;
    for (int i=0; i<nruns; ++i) err += pow(times[i]-mean, 2);
    err /= (nruns-1)*nruns; // variance of the mean
    err = sqrt(err); // standard error of the mean

    const std::pair<double, double> result(mean, err);
    return result;
}
