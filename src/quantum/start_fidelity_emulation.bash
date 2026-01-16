#!/bin/bash

for index in 0 1 2 3 4 5 6 7 8 9 #10 11 12 13 14 15 16 17 18 19
do
  nohup python -u src/quantum/fidelity_emulation.py --index=$index > logs/fidelity_emulation_log_$index.log &
done

for index in 0 1 2 3 4 5 6 7 8 9 #10 11 12 13 14 15 16 17 18 19
do
  nohup python -u src/quantum/fidelity_emulation.py --index=$index --diagonal > logs/fidelity_emulation_log_$index.log &
done