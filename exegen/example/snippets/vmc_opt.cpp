#include "vmc/EnergyMinimization.hpp"
#include "nfm/Adam.hpp"
#include "nfm/LogManager.hpp"

// -- Setup Adam optimization
nfm::Adam adam(psi.getNVP(), true /* use averaging to obtian final result */);
adam.setAlpha(0.02);
adam.setBeta1(0.5);
adam.setBeta2(0.9);
adam.setMaxNConstValues(20); // stop after 10 constant values (within error)
adam.setMaxNIterations(200); // limit to 200 iterations
if (myrank == 0) { nfm::LogManager::setLoggingOn(); } // normal logging on thread 0

// optimize the NNWF
if (myrank == 0) { cout << "   Optimization . . ." << endl; }
minimizeEnergy<EnergyGradientTargetFunction>(vmc, adam, E_NMC, G_NMC, false /*no gradient error*/, 0.001 /*regularization*/);
if (myrank == 0) { cout << "   . . . Done!" << endl << endl; }
