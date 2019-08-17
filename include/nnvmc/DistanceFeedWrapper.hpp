#ifndef NNVMC_DISTANCEFEEDWRAPPER_HPP
#define NNVMC_DISTANCEFEEDWRAPPER_HPP

#include "sannifa/Sannifa.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <exception>

namespace nnvmc_detail
{
double calc_distv(const int ndim, const double r1[], const double r2[], double distv[])
{
    for (int i = 0; i < ndim; ++i) { distv[i] = r2[i] - r1[i]; }
    return sqrt(std::inner_product(distv, distv + ndim, distv, 0.));
}
}; // nnvmc_detail

// Wrapper around sannifa(-compatible) network, providing the network with the set of all
// distances between the raw particle coordinates and to a passed vectors of static coordinates.
template <class ANNType>
class DistanceFeedWrapper final: public Sannifa
{
private:
    ANNType _rawNN;
    const int _nspacedim{};
    const int _nvecs{};
    const int _nstatic{};
    double * const _rstatic{};

    const int _ntotdim_vecs{};
    const int _ntotdim_static{};
    const int _ndists_vecs{};
    const int _ndists_static{};
    const int _ndists{};

    double * const _distvecs{};
    double * const _dists{};
    double * const _d1{};
    double * const _d2{};

    void _enableFirstDerivative() final { _rawNN.enableFirstDerivative(); }
    void _enableSecondDerivative() final { _rawNN.enableSecondDerivative(); }
    void _enableVariationalFirstDerivative() final { _rawNN.enableVariationalFirstDerivative(); }
    void _enableCrossFirstDerivative() final { throw std::runtime_error("CrossFirstDerivative not implemented in DistanceFeedWrapper."); }

    void _enableCrossSecondDerivative() final { throw std::runtime_error("CrossSecondDerivative not implemented in DistanceFeedWrapper."); }

    void _evaluate(const double in[], bool flag_deriv) final
    {
        // prepare distance feed

        int ipair = 0;
        // inter-vector distances
        for (int i = 0; i < _nvecs; ++i) {
            for (int j = i + 1; j < _nvecs; ++j) {
                _dists[ipair] = nnvmc_detail::calc_distv(_nspacedim, in + i*_nspacedim, in + j*_nspacedim, _distvecs + ipair*_nspacedim);
                ++ipair;
            }
        }
        // distances to static vectors
        for (int i = 0; i < _nvecs; ++i) {
            for (int j = 0; j < _nstatic; ++j) {
                _dists[ipair] = nnvmc_detail::calc_distv(_nspacedim, in + i*_nspacedim, _rstatic + j*_nspacedim, _distvecs + ipair*_nspacedim);
                ++ipair;
            }
        }

        // evaluate the internal NN
        _rawNN.evaluate(_dists, flag_deriv);

        // compute derivatives with respect to cartesian input
        if (flag_deriv) {
            std::fill(_d1, _d1 + this->getNOutput()*_ntotdim_vecs, 0.);
            std::fill(_d2, _d2 + this->getNOutput()*_ntotdim_vecs, 0.);

            if (this->hasFirstDerivative()) {
                const bool flag_d2 = this->hasSecondDerivative();
                for (int iout = 0; iout < _rawNN.getNOutput(); ++iout) {
                    ipair = 0;

                    // vec-vec derivative terms
                    for (int i = 0; i < _nvecs; ++i) {
                        const int dindex_i = iout*_ntotdim_vecs + i*_nspacedim;
                        for (int j = i + 1; j < _nvecs; ++j) {
                            const int dindex_j = iout*_ntotdim_vecs + j*_nspacedim;
                            const double rawD1 = _rawNN.getFirstDerivative(iout, ipair);
                            const double rawD2 = flag_d2 ? _rawNN.getSecondDerivative(iout, ipair) : 0.;

                            for (int k = 0; k < _nspacedim; ++k) {
                                const double distv_red = _distvecs[ipair*_nspacedim + k]/_dists[ipair];
                                _d1[dindex_i + k] -= rawD1*distv_red;
                                _d1[dindex_j + k] += rawD1*distv_red;
                                if (flag_d2) {
                                    const double dvr2 = distv_red*distv_red;
                                    _d2[dindex_i + k] += rawD2*dvr2 - rawD1*(dvr2 - 1.)/_dists[ipair];
                                    _d2[dindex_j + k] += rawD2*dvr2 - rawD1*(dvr2 - 1.)/_dists[ipair];
                                }
                            }
                            ++ipair;
                        }
                    }

                    // vec-static derivative terms
                    for (int i = 0; i < _nvecs; ++i) {
                        const int dindex_i = iout*_ntotdim_vecs + i*_nspacedim;
                        for (int j = 0; j < _nstatic; ++j) {
                            const double rawD1 = _rawNN.getFirstDerivative(iout, ipair);
                            const double rawD2 = flag_d2 ? _rawNN.getSecondDerivative(iout, ipair) : 0.;

                            for (int k = 0; k < _nspacedim; ++k) {
                                const double distv_red = _distvecs[ipair*_nspacedim + k]/_dists[ipair];
                                _d1[dindex_i + k] -= rawD1*distv_red;
                                if (flag_d2) {
                                    const double dvr2 = distv_red*distv_red;
                                    _d2[dindex_i + k] += rawD2*dvr2 - rawD1*(dvr2 - 1.)/_dists[ipair];
                                }
                            }
                            ++ipair;
                        }
                    }
                }
            }
        }
    }

public:
    // helper function for determining NN input size
    static constexpr int calcNDists(const int nvecs, const int nstatic) { return (nvecs*(nvecs-1))/2 + nvecs*nstatic; }

    // Construct / Deconstruct

    explicit DistanceFeedWrapper(const ANNType &rawNN, int nspacedim, int nvecs, int nstatic, const double rstatic[]):
            Sannifa(nvecs*nspacedim, rawNN.getNOutput(), rawNN.getNVariationalParameters(),
                    DerivativeOptions{rawNN.hasFirstDerivative(), rawNN.hasSecondDerivative(), rawNN.hasVariationalFirstDerivative(), false, false}),
            _rawNN(rawNN)/* copy-construct internal nn*/, _nspacedim(nspacedim), _nvecs(nvecs), _nstatic(nstatic), _rstatic(new double[nstatic*nspacedim]),
            _ntotdim_vecs(nspacedim*nvecs), _ntotdim_static(nspacedim*nstatic), _ndists_vecs((nvecs*(nvecs-1))/2), _ndists_static(nvecs*nstatic),
            _ndists(_ndists_vecs + _ndists_static), _distvecs(new double[_ndists*_nspacedim]), _dists(new double[_ndists]),
                    _d1(new double[rawNN.getNOutput()*_ntotdim_vecs]), _d2(new double[rawNN.getNOutput()*_ntotdim_vecs])
            {
                if (_rawNN.getNInput() != _ndists) {
                    throw std::invalid_argument("ANN number of inputs is not equal to the total number of distances (n_v*(n_v-1))/2 + n_v*n_s .");
                }
                std::copy(rstatic, rstatic + nstatic*nspacedim, _rstatic);
                std::fill(_distvecs, _distvecs + _ndists*_nspacedim, 0.);
                std::fill(_dists, _dists + _ndists, 0.);
                std::fill(_d1, _d1 + this->getNOutput()*_ntotdim_vecs, 0.);
                std::fill(_d2, _d2 + this->getNOutput()*_ntotdim_vecs, 0.);
            }

    DistanceFeedWrapper(const DistanceFeedWrapper &other): DistanceFeedWrapper(other._rawNN, other._nspacedim, other._nvecs, other._nstatic, other._rstatic) { } // copy construct

    ~DistanceFeedWrapper() final
    {
                delete [] _d2;
                delete [] _d1;
                delete [] _dists;
                delete [] _distvecs;
                delete [] _rstatic;
    }

    // wrapper-specific getters
    const ANNType &getRawNN(){ return _rawNN; }
    int getNSpaceDim() { return _nspacedim; }
    int getNVecs() { return _nvecs; }
    int getNStatic() { return _nstatic; }
    const double * getRStatic() { return _rstatic; }
    const double * getRStatic(int i_vec) { return _rstatic + i_vec*_nspacedim; }

    // Misc

    void saveToFile(const std::string &filename) const final { _rawNN.saveToFile(filename); } // ndim, nvecs, nstatic, rstatic not stored yet

    void printInfo(bool verbose) const final { _rawNN.printInfo(verbose); }
    std::string getLibName() const final {return "nnvmc/dists(" + _rawNN.getLibName() + ")"; }

    // Access

    double getVariationalParameter(int ivp) const final { return _rawNN.getVariationalParameter(ivp); }
    void getVariationalParameters(double vp[]) const final { _rawNN.getVariationalParameters(vp); }
    void setVariationalParameter(int ivp, double vp) final { _rawNN.setVariationalParameter(ivp, vp); }
    void setVariationalParameters(const double vp[]) final { _rawNN.setVariationalParameters(vp); }

    void getOutput(double out[]) const final { _rawNN.getOutput(out); }
    double getOutput(int i) const final { return _rawNN.getOutput(i); }

    void getFirstDerivative(double d1[]) const final { std::copy(_d1, _d1 + this->getNOutput()*_ntotdim_vecs, d1); }
    void getFirstDerivative(int iout, double d1[]) const final { std::copy(_d1 + iout*_ntotdim_vecs, _d1 + (iout + 1)*_ntotdim_vecs, d1); }
    double getFirstDerivative(int iout, int i1d) const final { return _d1[iout*_ntotdim_vecs + i1d]; }

    void getSecondDerivative(double d2[]) const final { std::copy(_d2, _d2 + this->getNOutput()*_ntotdim_vecs, d2); }
    void getSecondDerivative(int iout, double d2[]) const final { std::copy(_d2 + iout*_ntotdim_vecs, _d2 + (iout + 1)*_ntotdim_vecs, d2); }
    double getSecondDerivative(int iout, int i2d) const final { return _d2[iout*_ntotdim_vecs + i2d]; }

    void getVariationalFirstDerivative(double vd1[]) const final { _rawNN.getVariationalFirstDerivative(vd1); }
    void getVariationalFirstDerivative(int iout, double vd1[]) const final { _rawNN.getVariationalFirstDerivative(iout, vd1);  }
    double getVariationalFirstDerivative(int iout, int iv1d) const final { return _rawNN.getVariationalFirstDerivative(iout, iv1d); }

    void getCrossFirstDerivative(double d1vd1[]) const final {}
    void getCrossFirstDerivative(int iout, double d1vd1[]) const final {}
    double getCrossFirstDerivative(int iout, int i1d, int iv1d) const final { return 0.; }

    void getCrossSecondDerivative(double d2vd1[]) const final {}
    void getCrossSecondDerivative(int iout, double d2vd1[]) const final {}
    double getCrossSecondDerivative(int iout, int i2d, int iv1d) const final { return 0.; }
};


#endif