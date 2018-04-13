#!/bin/bash
# 1000 0.01 1 100000
for f in 0.01 10; do
	#statements
	for (( i = 1; i < 6; i++ )); do
		python supportnet_ec.py -s $i -l 1000 -f $f
	done
done

