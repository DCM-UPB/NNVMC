# LEGEND OF THE EXAMPLES

Make sure the examples are compiled, by running `make examples` in the project root folder.
Execute an example by switching into one of the example folders and running `./exe`.
Some examples might also contain a `plot.py` script to show a plot.
Run it after the exe by `python plot.py` (requires matplotlib).


## Example 1

`ex1/`: integration using a FeedForwardNeuralNetwork, making the integrals with and without MC


## Example 2

`ex2/`: another NN-MC integration example with beta derivatives calculation


## Example 3

`ex3/`: Optimize a FFNN trial wave function for 1-particle 1-dimension harmonic oscillator using the Conjugate Gradient method, setting the potential parameter w=1 and w=2.


## Example 4

`ex4/`: Compute the variational energy of Gaussian vs. NN fitted to Gaussian, for 1D1P harmonic oscillator


## Example 5

`ex5/`: solve 1 dimension 1 particle harmonic oscillator using a Neural Network Wave Function and optimizing via the Nelder-Mead Simplex algorithm.


## Example 6

`ex6/`: solve 1 dimension 1 particle harmonic oscillator using a Neural Network Wave Function and optimizing via the Adam algorithm.
