import time

from dataloader import BaseLoader
import os
import h5py
import pickle
import multiprocessing
import scipy as sp
import numpy as np
from concurrent_tools import MultiProcessPool
from geometry_tools import rotate_matrix
import transforms3d

DEFAULT_META_FILE = 'cmu_mocap.meta'
DEFAULT_DATA_CACHE_SUFFIX = '.dat'
DEFAULT_HIERARCHY_CACHE_SUFFIX = '.hier'

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


def parse_asf(file_path, cache_path, enable_cache):
    with open(file_path) as f:
        content = f.read().splitlines()

        in_bonedata = False
        in_hierarchy = False
        in_root = False

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
                        hierarchy = {}
                        # construct invert hierarchy list as: {child: [parent]}
                        # construct adjascent matrix
                        hierarchy['root'] = 'root' # root is the root of the tree
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

        print("parsed", file_path)
        hierarchy = CmuHierarchy(file_path.split('.')[0], joints, joint_idx, hierarchy, sp.sparse.coo_matrix(hierarchy_mat))
        if enable_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(hierarchy, f)
        return hierarchy


def parse_amc(file_path, cache_path=None, enable_cache=False):
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
        if enable_cache and cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(frames, f)
        print("parsed", file_path)
        return frames


def load_amc_cache(cache_file_path):
    with open(cache_file_path, 'rb') as f:
        print("loaded", cache_file_path)
        return pickle.load(f)


def load_asf_cache(cache_file_path):
    with open(cache_file_path, 'rb') as f:
        print("loaded", cache_file_path)
        return pickle.load(f)


class CmuLoader:

    """
    @param source_dir: the absolute directory of the asf and amc files
    @param cache_dir: the absolute directory of the cache files in pickle format
    @param enable_cache: if false, will load directly from the source, if true, will load from cache if exists
    """
    def __init__(self, source_dir, cache_dir=None, enable_cache=True):
        super(CmuLoader, self).__init__()
        self.task_pool = MultiProcessPool()

        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        self.source_dir = source_dir
        if enable_cache:
            if cache_dir:
                if not os.path.exists(cache_dir):
                    os.mkdir(cache_dir)
            else:
                self.enable_cache = False



    def load_meta(self):
        if self.enable_cache:
            meta_file_path = os.path.join(self.cache_dir, DEFAULT_META_FILE)
            if os.path.exists(meta_file_path):
                with open(meta_file_path, 'rb') as f:
                    meta = pickle.load(f)
                return meta

        meta = CmuMeta()

        for (root, dirs, files) in os.walk(self.source_dir, topdown=True):
            asf_name = None
            asf_path = None
            amc_names = []
            amc_paths = []
            for f in files:
                fname = f.split('.')[0]
                fsuffix = f.split('.')[-1]
                if fsuffix == 'amc':
                    amc_names.append(fname)
                    amc_path = os.path.join(root, f)
                    amc_paths.append(amc_path)
                elif fsuffix == 'asf':
                    asf_name = fname
                    asf_path = os.path.join(root, f)
            if asf_name and amc_names:
                for i, amc_name in enumerate(amc_names):
                    meta.mapping[amc_name] = [amc_name, amc_paths[i], asf_name, asf_path]
                    meta.amc_source_paths.append(amc_paths[i])
                    meta.amc_names.append(amc_name)
                meta.asf_names.append(asf_name)
                meta.asf_source_paths.append(asf_path)

        if self.enable_cache and self.cache_dir:
            meta_file_path = os.path.join(self.cache_dir, DEFAULT_META_FILE)
            with open(meta_file_path, 'wb') as f:
                pickle.dump(meta, f)
        return meta

    ''' ====================================
    load all data from the default directory structure of CMU dataset
    ========================================'''
    def load_all(self):

        # todo: remove cnt in release
        cnt = 0
        cnt_max = 100
        meta = self.load_meta()

        for amc_name in meta.amc_names:
            cnt +=1
            if cnt > cnt_max:
                break
            data_cache_path = os.path.join(self.cache_dir, amc_name + DEFAULT_DATA_CACHE_SUFFIX)
            data_source_path = meta.mapping[amc_name][1]
            if not os.path.exists(data_cache_path):
                self.task_pool.submit(tag='amc',
                                      task=parse_amc,
                                      params=(data_source_path, data_cache_path, self.enable_cache))
            else:
                self.task_pool.submit(tag='amc',
                                      task=load_amc_cache,
                                      params=(data_cache_path,))

        for i, fname in enumerate(meta.asf_names):
            cnt += 1
            if cnt > cnt_max:
                break
            cache_path = os.path.join(self.cache_dir, fname+DEFAULT_HIERARCHY_CACHE_SUFFIX)
            source_path = meta.asf_source_paths[i]
            if not os.path.exists(cache_path):
                self.task_pool.submit(tag='asf',
                                 task=parse_asf,
                                 params=(source_path, cache_path, self.enable_cache))
            else:
                self.task_pool.submit(tag='asf',
                                 task=load_asf_cache,
                                 params=(cache_path,))

        # wait for all tasks to finish
        self.task_pool.subscribe()
        # fetch results
        tic = time.time()
        data_raw = self.task_pool.fetch_results('amc')
        hierarchies = self.task_pool.fetch_results('asf')
        tac = time.time()
        print("fetching results takes", tac-tic, "seconds")

        self.task_pool.cleanup()

        # return raw data and hierarchy info
        return data_raw, hierarchies

    def load(self, amc_name, meta):
        amc_path = meta.mapping[amc_name][1]
        asf_name = meta.mapping[amc_name][2]
        asf_path = meta.mapping[amc_name][3]

        data_raw = None
        hierarchy = None
        if self.cache_dir:
            amc_cache_path = os.path.join(self.cache_dir, amc_name + DEFAULT_DATA_CACHE_SUFFIX)
            if not os.path.exists(amc_cache_path):
                data_raw = parse_amc(amc_path, amc_cache_path, self.enable_cache)
            else:
                data_raw = load_amc_cache(amc_cache_path)

            asf_cache_path = os.path.join(self.cache_dir, asf_name + DEFAULT_HIERARCHY_CACHE_SUFFIX)
            if not os.path.exists(asf_cache_path):
                hierarchy = parse_asf(asf_path, asf_cache_path, self.enable_cache)
            else:
                hierarchy = load_asf_cache(asf_cache_path)
        else:
            data_raw = parse_amc(amc_path, None, False)
            hierarchy = parse_asf(asf_path, None, False)

        return data_raw, hierarchy


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

        self.rmat = np.zeros((3,3))
        self.rmat_inv = np.zeros((3,3))
        self.calc_rmat()

    def calc_rmat(self):
        deg = np.deg2rad(self.axis)
        self.rmat = transforms3d.euler.euler2mat(*deg)
        # self.rmat = rotate_matrix(deg[2], deg[0], deg[1])
        self.rmat_inv = np.linalg.inv(self.rmat)

class CmuLabel:
    def __init__(self, labels):
        self.labels = labels

    # demo of segment label
    def segment(self):
        return [[[0, 100], ['run', 'squat']], [[100, 200], ['walk']]]

    def category(self):
        return ['run', 'walk', 'squat']

    def part_segment(self, part):
        return [[[0, 100], ['LA', 'RA'], ['write']]]


class CmuMeta:
    def __init__(self, amc_source_paths=[], asf_source_paths=[], mapping={}):
        self.amc_source_paths = amc_source_paths
        self.asf_source_paths = asf_source_paths
        self.mapping = mapping
        if len(asf_source_paths) > 0:
            self.amc_names = [os.path.basename(path) for path in amc_source_paths].split('.')[0]
        else:
            self.amc_names = []

        if len(asf_source_paths) > 0:
            self.asf_names = [os.path.basename(path) for path in asf_source_paths].split('.')[0]
        else:
            self.asf_names = []

    def find_hierarchy(self, data_name):
        return self.path_mapping[data_name]


class CmuHierarchy:
    def __init__(self, name, joints, joint_idx, hierarchy, hierarchy_mat):
        self.name = name
        self.joints = joints
        self.joint_idx = joint_idx

        #invert hierarchy list as: {child: [parent]}
        self.hierarchy = hierarchy
        self.hierarchy_mat = hierarchy_mat

        self.calc_order = {}

        self.gen_calc_order()

    def gen_calc_order(self):
        for jname in self.hierarchy.keys():
            order = [self.joint_idx[jname]]
            if jname == 'root':
                self.calc_order[jname] = order
                continue

            parent = self.hierarchy[jname]
            order.append(self.joint_idx[parent])
            while parent != 'root':
                parent = self.hierarchy[parent]
                order.append(self.joint_idx[parent])

            self.calc_order[jname] = order

    def get_parent(self, child):
        return self.hierarchy[child]