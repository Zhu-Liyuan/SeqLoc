from pathlib import Path
import parser
from matplotlib import scale
import numpy as np
from hloc.utils import read_write_model, parsers
import random
from pcr import teaser_pcr

import argparse


def parse_3d_pairs(path):
    pairs_3d = np.empty((0, 3), dtype=np.int32)
    scores_3d = np.empty(0)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            if len(p) == 0:
                continue
            q, r, prob = p.split()
            if int(q) == -1 or int(r) == -1: continue
            pairs_3d = np.append(pairs_3d, np.asarray([q, r], dtype=np.int32))
            scores_3d = np.append(scores_3d, np.asarray([prob], dtype=np.float64))

    return pairs_3d.reshape(-1, 2), scores_3d


## compute the scale factor between the ref point cloud and q point cloud

def scale_solver(pairs, scores, ref_pts_3d, q_pts_3d):
    # randomly sample pairs for several times and compute the distances between edges

    random_samples = 1000  ## a parameter that can be tuned

    grid_side = np.arange(len(pairs))
    all_possible_edges = np.array(np.meshgrid(grid_side, grid_side)).reshape(-1, 2)
    scales = []
    weights = []
    for i in range(random_samples):
        edge = random.choice(all_possible_edges)
        if edge[0] == edge[1]: continue

        pair1 = pairs[edge[0]]
        pair2 = pairs[edge[1]]
        q_xyz1 = q_pts_3d[pair1[0]].xyz
        q_xyz2 = q_pts_3d[pair2[0]].xyz
        ref_xyz1 = ref_pts_3d[pair1[1]].xyz
        ref_xyz2 = ref_pts_3d[pair2[1]].xyz
        q_dist = (np.linalg.norm(q_xyz1 - q_xyz2))
        ref_dist = (np.linalg.norm(ref_xyz1 - ref_xyz2))

        # update the weight for each obs.
        scales.append(ref_dist / q_dist)

        weights.append(np.linalg.norm([scores[edge]]))

    scales = np.asarray(scales, np.float64).reshape(-1)
    weights = np.asarray(weights, np.float64).reshape(-1)
    scale = np.average(scales, weights=weights)
    return scale


def fit_pcr(Y, X):
    ## fit a registration with several samples
    # zero mean, std 1
    factor = np.linalg.norm(Y)
    Y = np.asarray(Y) / factor
    X = np.asarray(X) / factor
    y0 = np.mean(Y, axis=0)
    x0 = np.mean(X, axis=0)
    H = np.zeros((3, 3), dtype=np.float64)
    for i in range(Y.shape[0]):
        H += (Y[i] - y0).T @ (X[i] - x0)

    U, D, VT = np.linalg.svd(H)
    R = (U @ VT).T
    t = y0 - R @ x0
    t *= factor
    residual = np.linalg.norm(Y.T - R @ X.T)

    return R, t, residual


def ransac_pcr(pairs_3d, scores, q_pts, ref_pts, scale, in_ratio=0.5, confidence=0.99, max_iters=100):
    # point cloud registration with known correspondences
    # compute the smallest number of iters that can satisfy the confidence
    N = np.log(1 - confidence) / np.log(1 - in_ratio ** 3)
    print(
        'To have a confidence of {} with the inlier ratio {}, we need at least {}'.format(confidence, in_ratio, int(N)))
    best_res = np.Infinity
    R0, t0 = 0, 0
    for i in range(int(N)):
        samples = random.sample(list(pairs_3d), 3)
        X = np.empty((3, 3))
        Y = np.empty((3, 3))
        for t in range(3):
            X = np.append(X, q_pts[samples[t][0]].xyz)
            Y = np.append(Y, ref_pts[samples[t][1]].xyz)
        X *= scale
        R, t, res = fit_pcr(Y.reshape(-1, 3), X.reshape(-1, 3))
        # print(R,t)
        if res < best_res:
            R0 = R
            t0 = t
            best_res = res
    return R0, t0


def write_3dpts_corr(pairs_3d, ref_pts, q_pts, pair_dir):
    with open(pair_dir, 'w') as f:
        for pair in pairs_3d:
            coor1 = ' '.join(map(str, q_pts[pair[0]].xyz))
            coor2 = ' '.join(map(str, ref_pts[pair[1]].xyz))
            f.write(f'{coor1} {coor2}\n')


def localizer(images, T, scale, output):
    camera_poses = []
    for image in images.items():
        # q_old
        qvec = image[1].qvec
        # R_old
        rmtx = read_write_model.qvec2rotmat(qvec)
        # Rotation from image -> local camera frame -> global pointcloud frame
        # R_new
        rmtx = T[:3, :3] / scale @ rmtx.T
        rmtx = rmtx.T
        # q_new
        qvec = read_write_model.rotmat2qvec(rmtx)
        # Translation from image -> local camera frame -> global pointcloud frame
        # t_new
        tvec = scale * image[1].tvec - rmtx @ T[:3, 3]
        camera_poses.append([image[1].name, qvec, tvec])

    if not output.exists(): output.touch()
    with open(output, 'w') as f:
        for pose in camera_poses:
            qvec = ' '.join(map(str, pose[1]))
            coor = ' '.join(map(str, pose[2]))
            f.write(f'{pose[0]} {qvec} {coor}\n')
    return camera_poses


def main(db_model, query_model, local_visual, sfm_path):
    ##  Path define
    if isinstance(db_model, str):
        db_model = Path(db_model)
    if isinstance(query_model, str):
        query_model = Path(query_model)

    pairs_3d_path = query_model / '3d_pairs.txt'
    # sfm_model = query_model/'outputs'/"sfm_superpoint+superglue/"
    # TODO
    sfm_model = Path(sfm_path) / 'outputs' / "sfm_superpoint+superglue/"

    # READ 3D PAIRS FILE
    pairs_3d, scores = parse_3d_pairs(pairs_3d_path)
    # READ SFM MODEL OF IMAGE SEQUENCE
    _, q_images, q_points3D = read_write_model.read_model(path=sfm_model, ext='.bin')
    # _, _, ref_points3D = read_write_model.read_model(path=db_model / "sfm_superpoint+superglue/", ext='.bin')
    # READ SFM MODEL OF WHOLE DATASET
    ref_points3D = read_write_model.read_points3D_binary(db_model / "sfm_superpoint+superglue/points3D.bin")
    # GET COORDINATES OF CORRESONDING POINTS IN QUERY AND REF
    corr_3d_path = query_model / "3d_corr.txt"
    write_3dpts_corr(pairs_3d, ref_points3D, q_points3D, corr_3d_path)
    query_pcd = query_model / 'outputs' / "point_cloud.ply"
    ref_pcd = db_model / 'outputs' / "point_cloud.ply"
    # COMPUTE TRANSFORMATION MATRIX: transform local pointcloud to global pointcloud
    # T: 4*4 transformation matrix
    T, t_scale = teaser_pcr.main(corr_3d_path, str(query_pcd), str(ref_pcd), VISUALIZE=local_visual)
    # LOCALIZE
    # CAMERA POSE UNDER GLOBAL POINT FRAME
    localizer(q_images, T, t_scale, query_model / 'localization_results.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate camera poses by point cloud registration')
    parser.add_argument('--local_visual', type=bool, default=False, help="Use open3d to visualize results")
    # TODO
    # parser.add_argument('--db_model', type=str, required=True, help="Path to the database model")
    # parser.add_argument('--query_model', type=str, required=True, help="Path to the query model")
    parser.add_argument('--db_model', type=str, default="/cluster/project/infk/courses/252-0579-00L/group16/output"
                                                        "/superpoint+superglue_aachen")
    parser.add_argument('--query_model', type=str, default="/cluster/project/infk/courses/252-0579-00L/group16/hs"
                                                           "/test/pairs_3d_from_2d/")
    parser.add_argument('--sfm_path', type=str,
                        default='/cluster/project/infk/courses/252-0579-00L/group16/hs/test/sfm_pipeline/')

    args = parser.parse_args()

    main(**args.__dict__)
