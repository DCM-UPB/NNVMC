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
//
// Experimental feature: Optionally sort the electronic coordinates according to some comparison function,
// before computing the distances and feeding to the NN. The intention is to create inherent exchange-symmetry.
// NOTE: Currently the sorting function is hard-coded, for testing.
template <class ANNType, bool useSort = false>
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

    int * const _idx_sorted{};
    double * const _in_sorted{};
    double * const _distvecs{};
    double * const _dists{};
    double * const _d1_dist{};
    double * const _d2_dist{};
    double * const _d1_orig{};
    double * const _d2_orig{};

    void _enableFirstDerivative() final { _rawNN.enableFirstDerivative(); }
    void _enableSecondDerivative() final { _rawNN.enableSecondDerivative(); }
    void _enableVariationalFirstDerivative() final { _rawNN.enableVariationalFirstDerivative(); }
    void _enableCrossFirstDerivative() final { throw std::runtime_error("CrossFirstDerivative not implemented in DistanceFeedWrapper."); }

    void _enableCrossSecondDerivative() final { throw std::runtime_error("CrossSecondDerivative not implemented in DistanceFeedWrapper."); }

    void _evaluateDerived(const double in[], const double orig_d1[], const double orig_d2[], bool flag_deriv) final {
        throw std::runtime_error("Non-original input feed not supported by DistanceFeedWrapper.");
    }

    void _evaluate(const double in_orig[], bool flag_deriv) final
    {
        const bool flag_d1 = flag_deriv && this->hasFirstDerivative();
        const bool flag_d2 = flag_deriv && this->hasSecondDerivative();
        if (flag_d1) { std::fill(_d1_dist, _d1_dist + _ndists*_ntotdim_vecs, 0.); }
        if (flag_d2) { std::fill(_d2_dist, _d2_dist + _ndists*_ntotdim_vecs, 0.); }

        if (useSort) { // prepare sorted input
            const bool verbose = false;

            if (verbose) {
                std::cout << "r_e input: ";
                for (int i = 0; i < _ntotdim_vecs; ++i) { std::cout << in_orig[i] << " "; }
                std::cout << std::endl;
            }
            // hard-coded potential energy calculation for testing
            double epot_ep[_nvecs];
            std::fill(epot_ep, epot_ep + _nvecs, 0.);
            for (int i = 0; i < _nvecs; ++i) {
                for (int j = 0; j < _nstatic; ++j) {
                    double distvec[_nspacedim]; // needed as argument, but isn't actually used here
                    const double dist = nnvmc_detail::calc_distv(_nspacedim, in_orig + i*_nspacedim, _rstatic + j*_nspacedim, distvec);
                    epot_ep[i] -= 1./dist;
                    if (verbose) { std::cout << "dist_ep " << i << "<->" << j << " : " << dist << std::endl; }
                }
                if (verbose) { std::cout << "epot_ep[" << i << "]: " << epot_ep[i] << std::endl; }
            }

            // setup arrays relevant to sorting and sort the input
            std::iota(_idx_sorted, _idx_sorted + _nvecs, 0); // fill a range 0...nvecs-1
            std::copy(in_orig, in_orig + _ntotdim_vecs, _in_sorted); // copy original input
            std::sort(_idx_sorted, _idx_sorted + _nvecs, [in = in_orig, ndim = _nspacedim, pot = &epot_ep[0]](int a, int b) {
                return pot[a] > pot[b]; // currently, sorting according to epot_e value
            });
            if (verbose) {
                std::cout << "Sorted indices: ";
                for (int i = 0; i < _nvecs; ++i) { std::cout << _idx_sorted[i] << " "; }
                std::cout << std::endl << std::endl;
            }
            for (int i = 0; i < _nvecs; ++i) { // fill sorted input array
                std::copy(in_orig + _idx_sorted[i]*_nspacedim, in_orig + (_idx_sorted[i] + 1)*_nspacedim, _in_sorted + i*_nspacedim);
            }
        }
        const double * const in = useSort ? _in_sorted : in_orig; // pointer to the relevant input


        // prepare distance feed incl. derivatives
        
        int ipair = 0;
        // inter-vector distances
        for (int i = 0; i < _nvecs; ++i) {
            for (int j = i + 1; j < _nvecs; ++j) {
                _dists[ipair] = nnvmc_detail::calc_distv(_nspacedim, in + i*_nspacedim, in + j*_nspacedim, _distvecs + ipair*_nspacedim);
                if (flag_d1) {
                    for (int k = 0; k < _nspacedim; ++k) {
                        const double distv_red = _distvecs[ipair*_nspacedim + k]/_dists[ipair];
                        _d1_dist[ipair*_ntotdim_vecs + i*_nspacedim + k] = -distv_red;
                        _d1_dist[ipair*_ntotdim_vecs + j*_nspacedim + k] = distv_red;
                        if (flag_d2) {
                            const double dvr2 = distv_red*distv_red;
                            _d2_dist[ipair*_ntotdim_vecs + i*_nspacedim + k] = - (dvr2 - 1.)/_dists[ipair];
                            _d2_dist[ipair*_ntotdim_vecs + j*_nspacedim + k] = - (dvr2 - 1.)/_dists[ipair];
                        }
                    }
                }
                ++ipair;
            }
        }
        // distances to static vectors
        for (int i = 0; i < _nvecs; ++i) {
            for (int j = 0; j < _nstatic; ++j) {
                _dists[ipair] = nnvmc_detail::calc_distv(_nspacedim, in + i*_nspacedim, _rstatic + j*_nspacedim, _distvecs + ipair*_nspacedim);
                if (flag_d1) {
                    for (int k = 0; k < _nspacedim; ++k) {
                        const double distv_red = _distvecs[ipair*_nspacedim + k]/_dists[ipair];
                        _d1_dist[ipair*_ntotdim_vecs + i*_nspacedim + k] = -distv_red;
                        if (flag_d2) {
                            _d2_dist[ipair*_ntotdim_vecs + i*_nspacedim + k] = - (distv_red*distv_red - 1.)/_dists[ipair];
                        }
                    }
                }
                ++ipair;
            }
        }

        // evaluate the internal NN (and inversely apply sorting to derivatives)
        if (flag_deriv) {
            _rawNN.evaluateDerived(_dists, _d1_dist, _d2_dist);
            if (useSort) {
                for (int i = 0; i < _nvecs; ++i) {
                    for (int j = 0; j < _nspacedim; ++j) {
                        if (flag_d1) { _d1_orig[i*_nspacedim + j] = _rawNN.getFirstDerivative(0, _idx_sorted[i]*_nspacedim + j); }
                        if (flag_d2) { _d2_orig[i*_nspacedim + j] = _rawNN.getSecondDerivative(0, _idx_sorted[i]*_nspacedim + j); }
                    }
                }
            }
            else {
                if (flag_d1) { _rawNN.getFirstDerivative(_d1_orig); }
                if (flag_d2) { _rawNN.getSecondDerivative(_d2_orig); }
            }
        }
        else { _rawNN.evaluateDerived(_dists); }
    }

public:
    // helper function for determining NN input size
    static constexpr int calcNDists(const int nvecs, const int nstatic) { return (nvecs*(nvecs-1))/2 + nvecs*nstatic; }

    // Construct / Deconstruct

    explicit DistanceFeedWrapper(const ANNType &rawNN, int nspacedim, int nvecs, int nstatic, const double rstatic[]):
            Sannifa(nvecs*nspacedim, nvecs*nspacedim, rawNN.getNOutput(), rawNN.getNVariationalParameters(),
                    DerivativeOptions{rawNN.hasFirstDerivative(), rawNN.hasSecondDerivative(), rawNN.hasVariationalFirstDerivative(), false, false}),
            _rawNN(rawNN)/* copy-construct internal nn*/, _nspacedim(nspacedim), _nvecs(nvecs), _nstatic(nstatic), _rstatic(new double[nstatic*nspacedim]),
            _ntotdim_vecs(nspacedim*nvecs), _ntotdim_static(nspacedim*nstatic), _ndists_vecs((nvecs*(nvecs-1))/2), _ndists_static(nvecs*nstatic),
            _ndists(_ndists_vecs + _ndists_static), _idx_sorted(new int[useSort ? nvecs : 0]), _in_sorted(new double[useSort ? nvecs*nspacedim : 0]),
            _distvecs(new double[_ndists*_nspacedim]), _dists(new double[_ndists]),
            _d1_dist(new double[_ndists*_ntotdim_vecs]), _d2_dist(new double[_ndists*_ntotdim_vecs]),
            _d1_orig(new double[_ndists*_ntotdim_vecs]), _d2_orig(new double[_ndists*_ntotdim_vecs])
            {
                if (_rawNN.getNInput() != _ndists) {
                    throw std::invalid_argument("ANN number of inputs is not equal to the total number of distances (n_v*(n_v-1))/2 + n_v*n_s .");
                }
                std::copy(rstatic, rstatic + _ntotdim_static, _rstatic);
                if (useSort) {
                    std::iota(_idx_sorted, _idx_sorted + _nvecs, 0);
                    std::fill(_in_sorted, _in_sorted + _ntotdim_vecs, 0.);
                }
                std::fill(_distvecs, _distvecs + _ndists*_nspacedim, 0.);
                std::fill(_dists, _dists + _ndists, 0.);
                std::fill(_d1_dist, _d1_dist + _ndists*_ntotdim_vecs, 0.);
                std::fill(_d2_dist, _d2_dist + _ndists*_ntotdim_vecs, 0.);
                std::fill(_d1_orig, _d1_orig + _ndists*_ntotdim_vecs, 0.);
                std::fill(_d2_orig, _d2_orig + _ndists*_ntotdim_vecs, 0.);
            }

    DistanceFeedWrapper(const DistanceFeedWrapper &other): DistanceFeedWrapper(other._rawNN, other._nspacedim, other._nvecs, other._nstatic, other._rstatic) { } // copy construct

    ~DistanceFeedWrapper() final
    {
                delete [] _d2_orig;
                delete [] _d1_orig;
                delete [] _d2_dist;
                delete [] _d1_dist;
                delete [] _dists;
                delete [] _distvecs;
                delete [] _in_sorted;
                delete [] _idx_sorted;
                delete [] _rstatic;
    }

    // wrapper-specific getters
    const ANNType &getRawNN() const { return _rawNN; }
    int getNSpaceDim() const { return _nspacedim; }
    int getNVecs() const { return _nvecs; }
    int getNStatic() const { return _nstatic; }
    const double * getRStatic() const { return _rstatic; }
    const double * getRStatic(int i_vec) const { return _rstatic + i_vec*_nspacedim; }

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

    /* // code from version without sorting
    void getFirstDerivative(double d1[]) const final { _rawNN.getFirstDerivative(d1); }
    void getFirstDerivative(int iout, double d1[]) const final { _rawNN.getFirstDerivative(iout, d1); }
    double getFirstDerivative(int iout, int i1d) const final { return _rawNN.getFirstDerivative(iout, i1d); }

    void getSecondDerivative(double d2[]) const final { _rawNN.getSecondDerivative(d2); }
    void getSecondDerivative(int iout, double d2[]) const final { _rawNN.getSecondDerivative(iout, d2); }
    double getSecondDerivative(int iout, int i2d) const final { return _rawNN.getSecondDerivative(iout, i2d); }
    */
    void getFirstDerivative(double d1[]) const final { std::copy(_d1_orig, _d1_orig + _ntotdim_vecs, d1); }
    void getFirstDerivative(int iout, double d1[]) const final { std::copy(_d1_orig, _d1_orig + _ntotdim_vecs, d1); }
    double getFirstDerivative(int iout, int i1d) const final { return _d1_orig[i1d]; }

    void getSecondDerivative(double d2[]) const final { std::copy(_d2_orig, _d2_orig + _ntotdim_vecs, d2); }
    void getSecondDerivative(int iout, double d2[]) const final { std::copy(_d2_orig, _d2_orig + _ntotdim_vecs, d2); }
    double getSecondDerivative(int iout, int i2d) const final { return _d2_orig[i2d]; }

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
