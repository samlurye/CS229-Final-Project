#!/bin/bash
for i in 0 1  2 3 4
do
   echo $i | python3 evolve-feedforward.py
done
