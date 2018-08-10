#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=48:00:00
#PBS -l gres=ccm
#PBS -N rgen
#PBS -q gpu
#PBS -V

cd ~/17s-cran/CIFAR10
module unload python
module load anaconda2
module load ccm
source activate 17s-cran-br2

# Must run like 'qsub rgen.sh -v F=shadow/*.hdf5,N=100


ccmrun ./rgen.py -f ${F} -n ${N} > rgen_`basename $F`.txt
