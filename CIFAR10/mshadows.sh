#!/bin/bash
IDX=0
while [ $IDX -lt $1 ]
do
    qsub shadowModels.sh -N shadowModels$IDX  -v N=1,S=$IDX
    IDX=$[$IDX+1]
done
