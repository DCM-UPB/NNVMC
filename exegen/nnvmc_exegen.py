#!/usr/bin/env python

import argparse
import yaml

parser = argparse.ArgumentParser(prog='nnvmc_exegen')
parser.add_argument('config',
                    help='Path of the config.yaml file, which contains the environmental configuration.')
parser.add_argument('setup', 
                    help='Path of the setup.yaml file, which describes the source file setup.')
parser.add_argument('-s', '--store', action='store_true',
                    help='Store the generated source file and compile script.')
args = parser.parse_args()
print(args)
parser.print_help()

with open(args.config) as c_file:
    with open(args.setup) as s_file:
        c = yaml.load(c_file, Loader=yaml.FullLoader)
        s = yaml.load(s_file, Loader=yaml.FullLoader)

        print(c)

        print(s['includes'])
        print(s['namespaces'])
        print(s['snippets'])

        with open('main.cpp', 'w') as o_file:
            # add global includes first
            for incl in s['includes']:
                o_file.write("#include <" + incl + ">\n")

            # open main function
            o_file.write("int main()\n{\n")

            # use "global" namespaces
            for nmspc in s['namespaces']:
                o_file.write("using namespace " + nmspc + ";\n")

            # init MPI
            o_file.write("const int myrank = MPIVMC::Init();\n")

            # fill in snippets
            # ...
            for snip in s['snippets']:
                with open(snip, 'r') as snip_file:
                    o_file.write(snip_file.read())

            # close MPI and main
            o_file.write("MPIVMC::Finalize();\n")
            o_file.write("return 0;\n}\n")
