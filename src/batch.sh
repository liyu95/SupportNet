#!/bin/bash
for l in 1000 0.01 1; do
	#statements
	for (( i = 1; i < 6; i++ )); do
		python nd_level_1.py -s $i -l $l
	done
done

