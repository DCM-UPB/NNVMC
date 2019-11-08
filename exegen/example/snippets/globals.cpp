#include "../../examples/common/ExampleFunctions.hpp"

#include "vmc/VMC.hpp"
#include "vmc/MPIVMC.hpp"
#include "vmc/EnergyMinimization.hpp"
#include "nfm/Adam.hpp"
#include "nfm/LogManager.hpp"
#include "nnvmc/SimpleNNWF.hpp"
#include "qnets/templ/TemplNet.hpp"
#include "qnets/actf/TanSig.hpp"
#include "qnets/actf/Exp.hpp"
#include "sannifa/QTemplWrapper.hpp"

using namespace vmc;
using namespace templ;

const long E_NMC = 1048576; // MC samplings to use for computing the initial/final energy
const long G_NMC = 32768; // MC samplings to use for computing the energy and gradient
double energy[4]; // energy
double d_energy[4]; // energy error bar

