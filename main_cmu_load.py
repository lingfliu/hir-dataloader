import os.path

import numpy as np
import scipy as sp

from dataloader import CmuLoader, CmuJoint

root_dir = os.path.join('D:\\data', 'cmu')
data_dir = os.path.join(root_dir, 'all_asfamc\\subjects')
cache_dir = os.path.join(root_dir, 'cache')

if __name__ == '__main__':

    cmuLoader = CmuLoader()

    data_raw, joint_names, hierarchy = cmuLoader.batch_load(root_dir, cache_dir, save_cache=False)