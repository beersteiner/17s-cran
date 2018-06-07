#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=00:01:00
#PBS -l gres=ccm
#PBS -N gputest
#PBS -q debug_gpu
#PBS -V
cd ~/17s-cran/
module load python
module load ccm
ccmrun ./test > test.out
