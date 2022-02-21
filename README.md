# Pre and post processing for EXESS 
Application for processing files for EXtreme-scale Electronic Structure System (EXESS) with a focus on lattice-energy type calculations using python.

## Python dependencies
- QCP (https://github.com/zoeseeger/qcp-python-app) and change sys.path.append(path_to_qcp)
- tqdm
- numpy
- pandas

## Functions
- Make separate dimer/trimer/tetramer calculations
- Make job from json files
- CSV from log files
- json to xyz\
An xyz file is created from an exess input file (json). The json can be given as a command line argument, otherwise the first json found in the working directory will be used.

- Make smaller shell from json\
Given a EXESS MBE input file (json) fragments whose center of mass lie within a given distance to the central fragment will be written to a new input file. The fragment IDs used to make the new input file are written to "indexes.txt". The json can be given as a command line argument, otherwise the first json found in the working directory will be used.

- xyz to json
Make an EXESS MBE or non-MBE exess input file from an xyz. 
