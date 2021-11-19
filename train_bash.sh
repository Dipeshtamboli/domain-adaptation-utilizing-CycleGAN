#!/bin/bash

domain1="amazon"
domain2="dslr"

# mv "datasets/cycle_office31/$domain1" "datasets/cycle_office31/trainA"
# mv "datasets/cycle_office31/${domain1}_test" "datasets/cycle_office31/trainA_test"

# mv "datasets/cycle_office31/$domain2" "datasets/cycle_office31/trainB"
# mv "datasets/cycle_office31/${domain2}_test" "datasets/cycle_office31/trainB_test"

python train.py --dataroot ./datasets/cycle_office31 --name AD_cyclegan --model cycle_gan


# mv "datasets/cycle_office31/trainA" "datasets/cycle_office31/${domain1}"
# mv "datasets/cycle_office31/trainA_test" "datasets/cycle_office31/${domain1}_test"

# mv "datasets/cycle_office31/trainB" "datasets/cycle_office31/${domain2}"
# mv "datasets/cycle_office31/trainB_test" "datasets/cycle_office31/${domain2}_test"
