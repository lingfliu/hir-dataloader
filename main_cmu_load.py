import os.path

import numpy as np
import scipy as sp

import time
from dataloader import CmuLoader, CmuJoint
from dataviewer import draw_hierarchy, draw_motion, draw_pos_motion
from cmu_data_slice import cmu_data_slice

source_dir = os.path.join('D:\\data', 'cmu')
data_dir = os.path.join(source_dir, 'all_asfamc\\subjects')
cache_dir = os.path.join(source_dir, 'cache')

if __name__ == '__main__':
    tic = time.time()
    cmuLoader = CmuLoader(source_dir, cache_dir, enable_cache=True)

    # 加载并缓存所有数据
    frames_list, pos_frames_list, hierarchy, meta = cmuLoader.load_all()

    # 加载某一个数据
    # meta = cmuLoader.load_meta()
    # hierarchy = None
    # pos_frames = None
    # frames = None
    # for idx, amc_name in enumerate(meta.amc_names):
    #     if idx == 120:
    #         frames, pos_frames, hierarchy = cmuLoader.load(amc_name, meta)
    #         break

    # 绘制关节连接图
    # draw_hierarchy(hierarchy)

    # 绘制运动序列
    # draw_motion(data_raw, hierarchy)
    # draw_pos_motion(pos_frames, hierarchy)

    # 数据切片并乱序加载
    frame_slices = cmu_data_slice(meta, 500, 50, False)
    pos_frames = []
    for frame_slice in frame_slices:
        pos_frame = pos_frames_list[frame_slice[0]][frame_slice[2]:frame_slice[3]]
        pos_frames.append(pos_frame)

    pos_frames = np.array(pos_frames)
    print(pos_frames.shape)

    kfold = 5

    tac = time.time()
    print('load time: ', tac - tic)
