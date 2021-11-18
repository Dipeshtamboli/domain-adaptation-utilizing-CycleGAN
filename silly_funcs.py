import os
def mkdir_func(dir_to_make):
    if not os.path.exists(dir_to_make): os.makedirs(dir_to_make)