#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=1:00:00
#PBS -l gres=ccm
#PBS -N exfil
#PBS -q gpu
#PBS -V

cd ~/17s-cran/CIFAR10
module unload python
module load anaconda2
module load ccm
source activate 17s-cran-br2

# Must run like 'qsub exfil.sh -v F=shadow/M00.hdf5,S=47405,I=0'
    # F is model file from which to exfil the data
    # S is seed to use
    # I is index of CIFAR10 image in order to create original

ccmrun ./exfil.py -f ${F} -s ${S} -i ${I} > exfil_out.txt
