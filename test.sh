#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=00:01:00
#PBS -N gputest
#PBS -q debug_gpu
#PBS -l gres=ccm
#PBS -V
module load python
module load ccm
ccmrun test
