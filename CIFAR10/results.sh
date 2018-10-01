#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=1:00:00
#PBS -l gres=ccm
#PBS -N results
#PBS -q gpu
#PBS -V

cd ~/17s-cran/CIFAR10
module unload python
module load anaconda2
module load ccm
source activate 17s-cran-br2

# Must run like 'qsub results.sh -v T=models/*.hdf5,A=atkmodels/atkmodel.hdf5'
    # T is target model file from which to determine maliciousness
    # a is attack model file we've built

ccmrun ./results.py -t ${T} -a ${A} > ./results/results_out.txt
