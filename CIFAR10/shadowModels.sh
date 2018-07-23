#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=48:00:00
#PBS -l gres=ccm
#PBS -N shadowModels
#PBS -q gpu
#PBS -V

cd ~/17s-cran/CIFAR10
module unload python
module load anaconda2
module load ccm
source activate 17s-cran-br2

# Must run like 'qsub shadowModels.sh -v N=3,S=21'
    # N is number of model pairs to generate
    # S is file number to start with

ccmrun ./shadow.py -e 200 -n ${N} -s ${S} > shadow`echo ${S}`.txt
