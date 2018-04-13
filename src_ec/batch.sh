#!/bin/bash
# 1000 0.01 1 100000
for l in 1000; do
	#statements
	for (( i = 1; i < 6; i++ )); do
		python supportnet_ec.py -s $i -l $l
	done
done

