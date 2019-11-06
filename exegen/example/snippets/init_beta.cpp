// create random generator
random_device rdev;
mt19937_64 rgen;
rgen = mt19937_64(rdev());
auto rd = normal_distribution<double>(0, sqrt(1./HIDDENLAYERSIZE)); // rule-of-thumb sigma

// setup initial weights
const int nvpar = ann.getNVariationalParameters();
double vp[nvpar];
if (myrank == 0) { // we need to make sure that all threads use the same initial weights
    int ivp_l1 = 0; // l1 weights running index
    int ivp_l2 = HIDDENLAYERSIZE*2; // l2 weights running index
    vp[ivp_l2] = 0.; // l2 bias init to 0
    for (int i = 0; i < HIDDENLAYERSIZE; ++i) {
        vp[ivp_l1] = 0.; // l1 bias init to 0
        vp[ivp_l1 + 1] = rd(rgen); // l1-l0 weights
        ivp_l1 += 2;
        vp[ivp_l2 + 1] = rd(rgen); // l2-l1 weights
        ++ivp_l2;
    }
    cout << "Init vp: ";
    for (double v : vp) { cout << v << " "; }
    cout << endl;
}
// broadcast the chosen weights to all threads
MPI_Bcast(vp, nvpar, MPI_DOUBLE, 0, MPI_COMM_WORLD);
ann.setVariationalParameters(vp);
