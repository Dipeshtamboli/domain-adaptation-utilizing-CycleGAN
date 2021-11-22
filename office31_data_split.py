import gdown
import os
import pdb
import glob
import shutil
import numpy as np
from silly_funcs import *

print("Downloading office-31 dataset from the official source")
url = "https://drive.google.com/u/0/uc?export=download&confirm=ogBi&id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE"
dir_to_download = "datasets/office31/"
filename = f"{dir_to_download}office31_dataset.tar.gz"
mkdir_func(dir_to_download)
gdown.download(url, filename, quiet = False)
print(f"File {filename} is downloaded.")

os.chdir(dir_to_download)
print(f"Current working directory: {os.getcwd()}")
print(f"Unpacking {filename.split('/')[-1]}")
os.system("tar -xf office31_dataset.tar.gz")

all_imgs = glob.glob('*/*/*/*.jpg')
os.chdir("../")
domains = ['amazon','dslr','webcam']
for domain in domains:
    mkdir_func(f"cycle_office31/{domain}")
    mkdir_func(f"cycle_office31/{domain}_test")

print(f"Restructuring Office-31 dataset in CycleGAN training format.")
for img_path in all_imgs:
    domain = img_path.split('/')[0]
    img_id = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]
    source = f"office31/{img_path}"
    # pdb.set_trace()
    if np.random.uniform(low=0.0, high=1.0, size=None) > 0.8:
        destination = f"cycle_office31/{domain}/{img_id}"
    else:
        destination = f"cycle_office31/{domain}_test/{img_id}"
    dest = shutil.copy(source, destination)