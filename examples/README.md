# LEGEND OF THE EXAMPLES

Make sure the examples are compiled, by running `./build.sh` in the project root folder.
Execute an example by switching into one of the example folders and running `./run.sh`.
Note that the actual example executables reside inside the `build/examples/` folder under the project's root.

IMPORTANT:
In the optimization examples, the FFNN are initialized completely randomly, which usually means that they
are extremely bad wave functions initially. Consequently, successful optimization depends a bit on luck.
Keep that in mind and restart the examples a few times if necessary.


## NNWF-VMC using QNets/Poly

`ex_vmc_polynet/`: Use a simple NNWF based on QNets/Poly to find harmonic oscillator ground state via VMC.


## NNWF-VMC using QNets/Templ

`ex_vmc_templnet/`: Like the previous example, but using QNets/Templ.


## NNWF pre-fitting

`ex_fit_basic/`: Pre-Fit the NNWF to a known solution/approximation before using it for VMC.


## Hydrogen Molecule

`ex_h2mol/`: Find the ground state energy of the hydrogen molecule with a simple bosonic product NNWF.