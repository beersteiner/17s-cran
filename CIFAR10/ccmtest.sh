#!/bin/bash


#PBS -l nodes=1:ppn=32
#PBS -l walltime=00:00:10
#PBS -q gpu
#PBS -l gres=ccm


module load ccm
ccmrun test.py > testout.txt
