#!/bin/bash

# Environment
module unload python
module load anaconda2
module load ccm
source activate 17s-cran-br2

# Directives
#PBS -l nodes=1:ppn=32
#PBS -l walltime=01:00:00
#PBS -q gpu
#PBS -l gres=ccm

# Execute
ccmrun cifar10-resnet.py > output.txt
