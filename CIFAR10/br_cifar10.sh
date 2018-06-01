#!/bin/bash

# scp -rf ./17s-cran jodstein@bigred2.uits.iu.edu:~/
# ssh jodstein@bigred2.uits.iu.edu
# git clone https://github.com/beersteiner/17s-cran.git
# cd 17s-cran/CIFAR10

module load ccm
module load anaconda2
#conda env create -f ../17s-crani-br2.yml
qsub -I -l walltime=01:00:00 -l nodes=1:ppn=16 -l gres=ccm -q gpu
module load ccm
ccmlogin
module unload python
module load anaconda2
source activate 17s-cran-br2

