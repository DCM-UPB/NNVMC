# NNVMC EXEGEN EXAMPLE

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
and `MPI_Init` / `MPI_Finalize` calls will be inserted for you, so do not repeat them. As usually, the MPI_Rank can be conveniently inquired
via the `myrank` const integer.

# Other hints
1) Remember that in C++ you are allowed to use `#include` lines anywhere in the code (but must be new line), so you do not have to include everything
via the global include list within the setup.yaml file. The same goes for the namespaces.

2) Creating code from snippets is a bit unconventional and requires some convention on naming for variables that are used in more than one snippet.
   Also, you have to maintain a correct order of the snippets within the setup.yaml file, whenever there is interdependency between the snippets.
   Concluding, this tool is best used only for the highest-level "scripty" part of an executable. Put everything else as proper functions into
   seperate files and `#include` them.

3) This tool is especially useful when working with TemplNet, which relies on a lot of static/compile-time configuration parameters.
   While keeping the rest of your application identical, you can easily switch the code blocks that deal with TemplNet creation. For example,
   you could have the `config.sh`, `setup.yaml` and common code in a directory, and then have a lot of subdirectories containing a lot of different
   TemplNet code snippets. Then just use the exegen script in every dir (should be easy with a tiny bash script) and you are ready to queue the runs on your cluster.
