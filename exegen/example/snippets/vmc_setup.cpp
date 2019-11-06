// Declare the trial wave function
SimpleNNWF<QTemplWrapper<NetType>> psi(1, 1, ann);

// Declare an Hamiltonian
// We use the harmonic oscillator with w=1 and w=2
const double w1 = 1.;
HarmonicOscillator1D1P ham(w1);

if (myrank == 0) { cout << "-> ham1:    w = " << w1 << endl << endl; }
VMC vmc(psi, ham); // VMC object used for energy/optimization

// set an integration range, because the NN might be completely delocalized
vmc.getMCI().setIRange(-7.5, 7.5);

// set fixed number of decorrelation steps
vmc.getMCI().setNdecorrelationSteps(1000);
