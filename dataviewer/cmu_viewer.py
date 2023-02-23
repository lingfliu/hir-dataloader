import math

import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import transforms3d

from geometry_tools import rotate_matrix


"""draw raw frame by converting to position"""
def draw_frame(axes, frame, hierarchy):
    pos_root = np.zeros((3, 1), dtype=np.float32)
    rmat_list = np.zeros((len(hierarchy.joints.keys()), 3, 3))
    offset_list = np.zeros((len(hierarchy.joints.keys()), 1, 3))

    idx_rec = []
    for jname in frame.keys():
        joint = hierarchy.joints[jname]
        jpos = frame[jname]
        axis = joint.axis
        if jname == 'root':
            pos_root = jpos[:3]
            deg = jpos[3:]
        else:
            deg = [0, 0, 0]
            cnt = 0
            for i, d in enumerate(joint.dof):
                if d > 0:
                    deg[i] = jpos[cnt]
                    cnt += 1

        # deg = np.deg2rad(deg) #np.array([deg])*math.pi/180
        deg = np.deg2rad(deg) #np.array([deg])*math.pi/180

        jidx = hierarchy.joint_idx[jname]
        idx_rec.append(jidx)

        # rmat = rotate_matrix(deg[2], deg[0], deg[1])
        rmat = transforms3d.euler.euler2mat(*deg)

        # amat = rotate_matrix(axis[2], axis[0], axis[1])
        amat = joint.rmat
        amat_inv = joint.rmat_inv

        # ref: https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html
        # rmat = amat.dot(rmat).dot(np.linalg.inv(amat))
        rmat = amat.dot(rmat).dot(amat_inv)

        rmat_list[jidx, :, :] = rmat

        direction = np.array([joint.direction])
        offset = joint.length*direction
        offset_list[jidx, :, :] = offset

    for i in range(hierarchy.joints.keys().__len__()):
        if i not in idx_rec:
            rmat_list[i, :, :] = np.eye(3, dtype=np.float32)

            jname = None
            for name in hierarchy.joint_idx.keys():
                if hierarchy.joint_idx[name] == i:
                    jname = name
                    break
            joint = hierarchy.joints[jname]
            direction = np.array([joint.direction])
            offset_list[i, :, :] = joint.length*direction

    pos = []
    for i, jname in enumerate(hierarchy.joints.keys()):
        rmat_parent = np.eye(3, dtype=np.float32)
        if jname == 'root':
            pos.append(np.transpose([pos_root]))
        else:
            calc_order = hierarchy.calc_order[jname]
            p = np.transpose([[0, 0, 0]])
            rmat_parent = rmat_list[calc_order[-1], :, :]
            offset = np.transpose(offset_list[calc_order[-1], :, :])
            p = np.transpose([pos_root]) + rmat_parent @ offset
            for idx in calc_order[::-1]:
                if idx == calc_order[-1]:
                    continue
                rmat = rmat_list[idx, :, :]
                offset = np.transpose(offset_list[idx])

                p = p + rmat_parent @ rmat @ offset
                rmat_parent = rmat_parent@rmat

            pos.append(p)
    pos = np.array(pos)


    # draw bone segments
    for jname in hierarchy.hierarchy.keys():
        parent_jname = hierarchy.hierarchy[jname]
        idx_j = hierarchy.joint_idx[jname]
        idx_p = hierarchy.joint_idx[parent_jname]
        axes.plot([pos[idx_j][0], pos[idx_p][0]],
                    [pos[idx_j][1], pos[idx_p][1]],
                    [pos[idx_j][2], pos[idx_p][2]],
                  'g', linewidth=1)

    # draw joints
    axes.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', marker='o', s=20)


    plt.show()


def draw_pos_frame(axes, pos, hierarchy):
    # draw bone segments
    for jname in hierarchy.hierarchy.keys():
        parent_jname = hierarchy.hierarchy[jname]
        idx_j = hierarchy.joint_idx[jname]
        idx_p = hierarchy.joint_idx[parent_jname]
        axes.plot([pos[idx_j][0], pos[idx_p][0]],
                  [pos[idx_j][1], pos[idx_p][1]],
                  [pos[idx_j][2], pos[idx_p][2]],
                  'g', linewidth=1)

    # draw joints
    axes.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', marker='o', s=20)


# draw motion by joint positions
def draw_pos_motion(pos_frames, hierarchy):
    plt.ion()
    axes = plt.axes(projection='3d')
    for pos in pos_frames:
        axes.clear()
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')
        axes.set_aspect('equal', adjustable='box')
        axes.set_zlim(-50, 50)
        axes.set_ylim(-10, 40)
        axes.set_xlim(-50, 50)
        axes.view_init(elev=144, azim=-83)

        draw_pos_frame(axes, pos, hierarchy)
        plt.pause(0.01)
        plt.show()


# draw motion from raw data (in degrees), loaded and converted into positions
def draw_motion(data_raw, hierarchy):
    plt.ion()
    axes = plt.axes(projection='3d')


    for frame in data_raw:
        axes.clear()
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')
        axes.set_aspect('equal', adjustable='box')
        axes.set_zlim(-50, 50)
        axes.set_ylim(-10, 40)
        axes.set_xlim(-50, 50)
        axes.view_init(elev=144, azim=-83)

        # draw each frame
        draw_frame(axes, frame[1], hierarchy)
        plt.pause(0.01)


def draw_hierarchy(hierarchy):
    # self.name = name
    # self.joints = joints
    # self.joint_idx = joint_idx
    # self.hierarchy = hierarchy
    # self.hierarchy_mat = hierarchy_mat
    axes = plt.axes(projection='3d')

    pos = []
    rmat_list = []
    offset_list = []
    for i, jname in enumerate(hierarchy.joints.keys()):
        joint = hierarchy.joints[jname]

        offset = np.zeros((3,1), dtype=np.float32)

        # direction+length: the offset of the joint at its local coordinate
        direction = np.array([joint.direction])
        # print(np.linalg.norm(direction))
        # direction = direction / np.linalg.norm(direction)
        length = joint.length

        # axis: the rotation of the joint at its parent coordinate
        axis = np.array(joint.axis)
        axis = axis / math.pi * 180

        rmat = rotate_matrix(axis[2], axis[0], axis[1])
        offset = length*direction

        rmat_list.append(rmat)
        offset_list.append(offset)

    for i, jname in enumerate(hierarchy.joints.keys()):
        if jname == 'root':
            pos.append(np.zeros((3,1)))
            continue

        calc_order = hierarchy.calc_order[jname]
        p = np.transpose([[0,0,0]])
        rmat_parent = rmat_list[calc_order[-1]]
        offset = np.transpose(offset_list[calc_order[-1]])
        p = p + rmat_parent@offset
        for idx in calc_order[::-1]:
            if idx == calc_order[-1]:
                continue
            print(idx)
            rmat = rmat_list[idx]
            offset = np.transpose(offset_list[idx])

            p = p + rmat_parent@offset
            # rmat_parent = rmat@rmat_parent


        pos.append(p)
    pos = np.array(pos)

    # draw bone segments
    for jname in hierarchy.hierarchy.keys():
        parent_jname = hierarchy.hierarchy[jname]
        idx_j = hierarchy.joint_idx[jname]
        idx_p = hierarchy.joint_idx[parent_jname]
        axes.plot([pos[idx_j][0], pos[idx_p][0]],
                    [pos[idx_j][1], pos[idx_p][1]],
                    [pos[idx_j][2], pos[idx_p][2]],
                  'g', linewidth=1)

    # draw joints
    axes.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', marker='o', s=20)

    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    axes.set_aspect('equal', adjustable='box')
    plt.show()





