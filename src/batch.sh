#!/bin/bash
for (( i = 1; i < 6; i++ )); do
	python nd_level_1.py -s $i
done