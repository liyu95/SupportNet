#!/bin/bash
# 100 10000
for f in 0.001 100; do
	#statements
	for (( i = 1; i < 6; i++ )); do
		python supportnet_ec.py -s $i -l 1000 -f $f
	done
done