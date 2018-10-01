#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=48:00:00
#PBS -l gres=ccm
#PBS -N atk
#PBS -q gpu
#PBS -V

cd ~/17s-cran/CIFAR10
module unload python
module load anaconda2
module load ccm
source activate 17s-cran-br2

# Must run like 'qsub atk.sh -v F=attack/consolidated.csv'


ccmrun ./atk.py -f ${F} > atk_`basename $F`.txt
