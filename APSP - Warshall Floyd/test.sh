#!/bin/bash
for p in 0.33 0.45 0.66
do
for q in 7 8 9 10 11 12
do
./apsp_cuda $q $p >> results_3.txt
done
done
