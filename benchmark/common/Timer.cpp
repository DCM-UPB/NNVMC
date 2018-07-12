#include <iostream>
#include <chrono>

class Timer
{
public:
    explicit Timer(const double scale = 1.) : _beg(_clock::now()), _scale(scale) {}
    void reset() { _beg = _clock::now(); }
    double elapsed() const {
        return _scale * std::chrono::duration_cast<_second>(_clock::now() - _beg).count(); }

private:
    typedef std::chrono::high_resolution_clock _clock;
    std::chrono::time_point<_clock> _beg;
    const double _scale;
    typedef std::chrono::duration<double, std::ratio<1>> _second;
};
