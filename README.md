# Pre and post processing for EXESS 
Application for processing files for EXtreme-scale Electronic Structure System (EXESS) with a focus on lattice-energy type calculations using python.

## Python dependencies
- QCP (https://github.com/zoeseeger/qcp-python-app) and change sys.path.append(path_to_qcp)
- tqdm
- numpy
- pandas
- h5py (for reading from a restart file)

## Functions
- Make separate dimer/trimer/tetramer calculations\
Separates a json mbe calculation into single dimer, trimer and tetramer calculations. 

- Make job from json files\
Makes Gadi scripts for the json files in the current directory in batches determined by the user.
- CSV from log files\
Makes a csv from the ouput of an exess fragmentation calculation. If folder 'dimers' exists, it will attempt to get the
energies from separate log files existing in folders 'dimers', 'trimers' and 'tetramers' as per the 
'Make separate dimer/trimer/tetramer calculations' function of this script. If the file 'frag_ids.txt'
exists - it is expected to contain all of the 'grep "<-" *log' lines of a run - the script will get the energies from 
the restart file ending in '*h5'. Else, the script will read from the end of a log file in the current directory.
- json to xyz\
An xyz file is created from an exess input file (json). The json can be given as a command line argument, otherwise the first json found in the working directory will be used.

- Make smaller shell from json\
Given a EXESS MBE input file (json) fragments whose center of mass lie within a given distance to the central fragment will be written to a new input file. The fragment IDs used to make the new input file are written to "indexes.txt". The json can be given as a command line argument, otherwise the first json found in the working directory will be used.

- xyz to json\
Make an EXESS MBE or non-MBE exess input file from an xyz. 

- Get geometry from frag list\
Will make a full ab initio file for a list of monomers supplied.  