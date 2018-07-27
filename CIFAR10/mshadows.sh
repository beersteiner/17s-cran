#!/bin/bash
# Usage: mshadow <startnum> <endnum>
IDX=$1
while [ $IDX -lt $2 ]
do
    qsub shadowModels.sh -N shadowModels$IDX  -v N=$( expr $2 - $1 ),S=$IDX
    IDX=$[$IDX+1]
done
