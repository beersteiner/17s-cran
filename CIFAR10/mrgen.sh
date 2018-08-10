#!/bin/bash
# Usage: ./mrgen -n <num_batches> -d <directory>

while getopts ":n:d:" opt; do
    case ${opt} in
        n)
            NBAT=$OPTARG
            ;;
        d)
            DIR=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$opt" >&2
            exit 1
            ;;
        :)
            echo "Option -$opt requires an argument." >&2
    esac
done

#echo "-n given as $NBAT"
#echo "-d given as $DIR"

if ! [[ $NBAT =~ [0-9]+ ]] || [ -z "$DIR" ]; then
    echo "Usage mrgen.sh -n <num_batches> -d <directory>"
    exit 1
fi


for file in $DIR/*.hdf5; do
    qsub rgen.sh -N rgen_`basename $file` -v F=$file,N=$NBAT
done
