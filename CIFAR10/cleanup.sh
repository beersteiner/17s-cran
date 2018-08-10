#!/bin/bash
rm out_good.txt
rm training_good.log
rm out_bad.txt
rm training_bad.log
rm shadow*.txt
rm atk_*.txt
find . -maxdepth 1 -type f -regex '.*\.[eo][0-9]+' -delete
