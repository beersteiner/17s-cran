#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=36:00:00
#PBS -l gres=ccm
#PBS -N goodModel
#PBS -q gpu
#PBS -V

cd ~/17s-cran/CIFAR10
module unload python
module load anaconda2
module load ccm
source activate 17s-cran-br2
ccmrun ./cifar10good.py -e 200 > out_good.txt
