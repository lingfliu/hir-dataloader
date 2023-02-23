import os.path

import numpy as np
import scipy as sp

import time
from dataloader import CmuLoader, CmuJoint
from dataviewer import draw_hierarchy, draw_motion, draw_pos_motion

source_dir = os.path.join('D:\\data', 'cmu')
data_dir = os.path.join(source_dir, 'all_asfamc\\subjects')
cache_dir = None # os.path.join(source_dir, 'cache')

if __name__ == '__main__':

    tic =time.time()
    cmuLoader = CmuLoader(source_dir, cache_dir, enable_cache=False)
    data_raw, hierarchy = cmuLoader.load_all()
    #
    # meta = cmuLoader.load_meta()
    #
    # hierarchy = None
    # pos_frames = None
    # frames = None
    # for idx, amc_name in enumerate(meta.amc_names):
    #     if idx == 120:
    #         frames, pos_frames, hierarchy = cmuLoader.load(amc_name, meta)
    #         break
    # # draw_hierarchy(hierarchy)
    # # draw_motion(data_raw, hierarchy)
    # draw_pos_motion(pos_frames, hierarchy)
    tac = time.time()
    print('load time: ', tac - tic)
