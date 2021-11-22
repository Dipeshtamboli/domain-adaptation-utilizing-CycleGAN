#!/bin/bash

declare -a arr=("ae_photos" "apple2orange" "summer2winter_yosemite" "horse2zebra" "monet2photo" "cezanne2photo" "ukiyoe2photo" "vangogh2photo" "maps" "cityscapes" "facades" "iphone2dslr_flower" "mini" "mini_pix2pix" "mini_colorization")

for dataset in "${arr[@]}"
do
    bash ./datasets/download_cyclegan_dataset.sh "$dataset"
done