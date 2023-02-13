from dataloader import BaseLoader
import os
import h5py
import multiprocessing
import scipy as sp
import numpy as np
from concurrent_tools import MultiProcessPool

DEFAULT_CACHE_FILE = 'cmu_cache.h5'


def amc_frame2bvh_frame(frame, joint_idx, joint_axis, joint_dof):
    frame_pos = []
    for j in frame.keys():
        idx = joint_idx[j]
        axis = joint_axis
        dof = joint_dof[idx]
        rot = [0, 0, 0]
        for i, r in enumerate(frame[j]):
            rot[i] = r

        frame_pos.append([axis, rot])
    return frame_pos


def hierarchy2adj(hierarchy):
    adj = np.zeros((len(hierarchy), len(hierarchy)))
    for i, h in enumerate(hierarchy):
        for j in h:
            adj[i, j] = 1
    return adj


def parse_asf(file_path):
    with open(file_path) as f:
        content = f.read().splitlines()

        in_bonedata = False
        in_hierarchy = False
        in_root = False
        idx = 0

        root_lines = []
        joint_lines = []
        hierarchy_lines = []
        is_begin = False
        joints = {}
        hierarchy = {}
        joint_idx = {}

        def parse_root(lines):
            name = 'root'
            axis = [float(x) for x in lines[2].split()[1:]]
            direction = [float(x) for x in lines[3].split()[1:]]

            root = CmuJoint(name, axis, direction, 0)
            return root

        for idx, line in enumerate(content):
            if line.strip() == ':root':
                in_root = True
                in_bonedata = False
                in_hierarchy = False

                continue

            if line.strip() == ':bonedata':
                in_root = False
                in_bonedata = True
                in_hierarchy = False
                is_begin = False
                if 'root' not in joints.keys():
                    joint_root = parse_root(root_lines)
                    joints['root'] = joint_root


            elif line.strip() == ':hierarchy':
                in_root = False
                in_bonedata = False
                in_hierarchy = True
                is_begin = False

                if 'root' not in joints.keys():
                    joint_root = parse_root(root_lines)
                    joints['root'] = joint_root

                # generate joint_idx
                for i, j in enumerate(joints.keys()):
                    joint_idx[j] = i

                continue
            else:
                # searching the begin marker
                if line.strip() == 'begin':
                    is_begin = True
                    if in_bonedata:
                        joint_lines = []
                    elif in_hierarchy:
                        hierarchy_lines = []
                    continue

                # when end marker is found, parse the data
                elif line.strip() == 'end':
                    is_begin = False
                    if in_bonedata:
                        # convert joint_lines to joint
                        # id = joint_lines[0].split()[1] # not used
                        name = joint_lines[1].split()[1]
                        direction = [float(x) for x in joint_lines[2].split()[1:]]
                        length = float(joint_lines[3].split()[1])
                        axis = [float(x) for x in joint_lines[4].split()[1:4]]

                        dof = [0, 0, 0]
                        limits = [[0, 0, 0]]*3
                        if len(joint_lines) > 5: # dof defined
                            dof_names = [x for x in joint_lines[5].split()[1:]]
                            if 'rx' in dof_names:
                                dof[0] = 1
                            if 'ry' in dof_names:
                                dof[1] = 1
                            if 'rz' in dof_names:
                                dof[2] = 1

                            cnt = 0
                            for i, d in enumerate(dof):
                                if d == 1:
                                    limits[i] = (joint_lines[6 + cnt].split('(')[1].split(')')[0].split(','))
                                    cnt += 1

                        joints[name] = CmuJoint(name, axis, direction, length, dof, limits)

                    elif in_hierarchy:
                        # construct invert hierarchy list as: {child: [parent]}
                        hierarchy = {}
                        # construct adjascent matrix
                        hierarchy_mat = np.zeros((len(joint_idx), len(joint_idx)))
                        for line in hierarchy_lines:
                            names = line.split()
                            for name in names[1:]:
                                hierarchy[name] = names[0]

                            root_idx = joint_idx[names[0]]
                            for name in names:
                                hierarchy_mat[root_idx, joint_idx[name]] = 1
                else:

                    if is_begin and in_bonedata:
                        joint_lines.append(line)
                    elif is_begin and in_hierarchy:
                        hierarchy_lines.append(line)
                    elif in_root:
                        root_lines.append(line)

        return file_path.split('.')[0], joints, joint_idx, hierarchy, sp.sparse.coo_matrix(hierarchy_mat)


def parse_amc(file_path):
    with open(file_path) as f:
        content = f.read().splitlines()
        is_data = False
        frames = []
        idx_frame = 0
        degrees = {}
        first_frame = True

        for idx, line in enumerate(content):
            if line == ':DEGREES':
                is_data = True
                continue
            else:
                if is_data:
                    data = line.split()
                    if data[0].isnumeric():
                        # reset frame buff
                        if first_frame:
                            first_frame = False
                            idx_frame = data[0]
                            continue
                        frames.append((idx_frame, degrees))
                        idx_frame = data[0]
                        degrees = {}
                        continue
                    else:
                        degrees[data[0]] = [float(deg) for deg in data[1:]]
        return frames


class CmuLoader(BaseLoader):

    def __init__(self):
        super(CmuLoader, self).__init__()
        self.task_pool = MultiProcessPool()

    def batch_load(self, data_dir, cache_dir=None, save_cache=False):
        if save_cache:
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)

        cache_file = os.path.join(cache_dir, DEFAULT_CACHE_FILE)
        if os.path.exists(cache_file):
            with h5py.File(os.path.join(cache_file, DEFAULT_CACHE_FILE), 'r') as h5f:
                data_raw = h5f['data_raw']
                hierarchy = h5f['hierarchy']
                joint_names = h5f['joint_names']

                return data_raw, joint_names, hierarchy
        else:
            '''walk through the directory and print the path of each file'''

            for (root, dirs, files) in os.walk(data_dir, topdown=True):
                for f in files:
                    if f.split('.')[-1] == 'amc':
                        amc_path = os.path.join(root, f)
                        # print('parsing amc %s' % amc_path)
                        self.task_pool.submit(tag='amc', task=parse_amc, args=(amc_path,))
                        pass
                    elif f.split('.')[-1] == 'asf':
                        asf_path = os.path.join(root, f)
                        # print('parsing asf %s' % asf_path)
                        self.task_pool.submit(tag='asf', task=parse_asf, args=(asf_path,))

            self.task_pool.subscribe()
            data = self.task_pool.fetch_results('amc')
            print(len(data))
            hierarchy = self.task_pool.fetch_results('asf')
            self.task_pool.cleanup()

            if save_cache:
                # todo: save cache in .h5 file
                pass
            else:
                return data, hierarchy


class CmuJoint:
    def __init__(self, name, axis, direction, length, dof=None, limits=None):
        if limits is None:
            limits = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        if dof is None:
            dof = [0, 0, 0]

        self.axis = axis
        self.name = name
        self.limits = limits
        self.dof = dof
        self.direction = direction
        self.length = length
