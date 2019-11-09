# NNVMC EXEGEN TOOL

nnvmc_exegen.py is small tool to help you to directly generate NNVMC executables from config files and source code snippets.
In the following we provide some basic information about the required files, but otherwise the example should be self-explanatory.

# Command Line Usage
Execute the nnvmc_exegen.py script as follows:
`./path_to_script.py config.sh setup.yaml`

To see usage hints on the command line use
`./path_to_script.py --help`


# config.sh
The required "config.sh" file provides environmental information for the build (compiler flags, library locations, etc.)
and follows the same scheme as it does in all NNVMC-related repositories. A template is provided in "config_template.sh".

# setup.yaml
In this file you define how the executable's source file should be generated. You may configure some global includes and namespaces
and otherwise provide the code blocks within the main function via the snippet file list. Code for opening/closing of the main function
and `MPI_Init` / `MPI_Finalize` calls will be inserted for you, so do not repeat them. This implies that the MPIVMC.hpp header is always
included as well. As usually, the MPI_Rank can be conveniently inquired via the `myrank` const integer.

# Output
If the build is successful, you will find a newly created `build` directory with the executable file in it.

# Other hints
1) You can have `#include` lines anywhere in your code snippets, but all includes will be concatenated together and printed at the beginning
   of the generated source file.

2) `using namespace ...` statements within the snippets will be left in place, but remember that all (and only the) following code snippets will
    be affected by this.

3) Creating code from snippets may be a bit untypical and requires some convention on naming for variables that are used in more than one snippet.
   Also, you have to maintain a correct order of the snippets within the setup.yaml file, whenever there is interdependency between the snippets.
   Concluding, this tool is best used only for the highest-level "scripty" part of an executable. Put everything else as proper functions into
   separate files and `#include` them.

4) This tool is especially useful when working with TemplNet, which relies on a lot of static/compile-time configuration parameters.
   While keeping the rest of your application identical, you can easily switch the code blocks that deal with TemplNet creation. For example,
   you could have the `config.sh`, `setup.yaml` and common code in a directory, and then have a lot of subdirectories containing a lot of different
   TemplNet code snippets. Then just use the exegen script in every dir (should be easy with a tiny bash script) and you are ready to queue the runs on your cluster.
