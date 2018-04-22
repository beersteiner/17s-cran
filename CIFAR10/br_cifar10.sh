#!/bin/bash

module load anaconda2
conda create -f ../17s-cran.yml
qsub -I -l walltime=01:00:00 -l nodes=1:ppn=16 -q debug_gpu
