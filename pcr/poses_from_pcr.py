from pathlib import Path
from nbformat import read
import numpy as np
from hloc.utils import read_write_model, parsers, io
import random
from pcr import teaser_pcr
import pycolmap
import argparse
from collections import defaultdict
from tqdm import tqdm

def parse_3d_pairs(path):
    pairs_3d = np.empty((0,3),dtype=np.int32)
    scores_3d = np.empty(0)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            if len(p) == 0:
                continue
            q, r, prob = p.split()
            if int(q) == -1 or int(r) == -1: continue
            pairs_3d = np.append(pairs_3d, np.asarray([q,r], dtype=np.int32))
            scores_3d = np.append(scores_3d, np.asarray([prob], dtype=np.float64))
    
    return pairs_3d.reshape(-1,2), scores_3d

## compute the scale factor between the ref point cloud and q point cloud

def scale_solver(pairs, scores, ref_pts_3d, q_pts_3d):
    # randomly sample pairs for several times and compute the distances between edges
    
    random_samples = 1000 ## a parameter that can be tuned
    
    grid_side = np.arange(len(pairs))
    all_possible_edges = np.array(np.meshgrid(grid_side,grid_side)).reshape(-1, 2)
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
        
        #update the weight for each obs.
        scales.append(ref_dist / q_dist)
        
        weights.append(np.linalg.norm([scores[edge]]))
    
    scales = np.asarray(scales, np.float64).reshape(-1)
    weights = np.asarray(weights, np.float64).reshape(-1)
    scale = np.average(scales, weights = weights)
    return scale
        
        
def fit_pcr(Y, X):
    ## fit a registration with several samples
    # zero mean, std 1
    factor = np.linalg.norm(Y)
    Y = np.asarray(Y) / factor
    X = np.asarray(X) / factor
    y0 = np.mean(Y, axis = 0)
    x0 = np.mean(X, axis = 0)
    H = np.zeros((3,3), dtype = np.float64)
    for i in range(Y.shape[0]):
        H += (Y[i] - y0).T @ (X[i] - x0)
    
    U, D, VT = np.linalg.svd(H)
    R = (U @ VT).T
    t = y0 - R @ x0
    t *= factor
    residual = np.linalg.norm(Y.T - R @ X.T)
    
    return R, t, residual
    
def ransac_pcr(pairs_3d, scores, q_pts, ref_pts, scale, in_ratio = 0.5, confidence = 0.99, max_iters = 100):
    # point cloud registration with known correspondences
    # compute the smallest number of iters that can satisfy the confidence
    N = np.log(1 - confidence) / np.log(1 - in_ratio ** 3)
    print('To have a confidence of {} with the inlier ratio {}, we need at least {}'.format(confidence, in_ratio, int(N)))
    best_res = np.Infinity
    R0,t0 = 0, 0
    for i in range(int(N)):
        samples = random.sample(list(pairs_3d), 3)
        X = np.empty((3,3))
        Y = np.empty((3,3))
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
            
def rotation_averaging():
    raise NotImplementedError

def localizer(images, T, scale, output):
    camera_poses = []
    # scale = np.linalg.norm(T[:3,:3])
    for image in images.items():
        qvec = image[1].qvec # camera_qvec in local point cloud
        camera_rmtx = read_write_model.qvec2rotmat(qvec).T
        camera_coor = - (camera_rmtx @ image[1].tvec).reshape(3,1) * scale
        local_projection_center = np.append(camera_rmtx, camera_coor, axis=1)
        local_projection_mtx = np.append(local_projection_center, np.array([0,0,0,1]).reshape(1,4), axis = 0)
        world_projection_mtx = T @ local_projection_mtx 
        world_rotmtx = world_projection_mtx[:3,:3]
        qvec = read_write_model.rotmat2qvec(world_rotmtx.T)
        tvec = - world_rotmtx.T @ world_projection_mtx[0:3,3]
        camera_poses.append([image[1].name, qvec, tvec])
    
    if not output.exists(): output.touch()
    with open(output, 'w') as f:
        for pose in camera_poses:
            qvec = ' '.join(map(str, pose[1]))
            coor = ' '.join(map(str, pose[2]))
            f.write(f'{pose[0]} {qvec} {coor}\n')
    return camera_poses

def pose_refiner(
        pcr_results,
        features_path,
        matches_path,
        retrievel_path,
        query_sfm,
        reference_sfm, 
        output
    ):
    """refine the pose of query images one by one
    by calling pycolmap.pose_refinement()
    codebase: hloc/localize_sfm.py
    Input args:
        pcr_results,
        q_cameras, 
        q_images,
        features_path,
        matches_path
    Output args:
        Camera_poses:[qname, qvec, tvec]

    """
    pcr_images = {}
    reference_sfm = pycolmap.Reconstruction(reference_sfm)
    query_sfm = pycolmap.Reconstruction(query_sfm)
    q_images = query_sfm.images
    q_cameras = query_sfm.cameras
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}
    with open(pcr_results, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            image = p.split()
            qname = image[0]
            qcamera = None
            qvec = np.asarray(image[1:5], dtype=np.float64)
            tvec = np.asarray(image[5:], dtype=np.float64)
            for q_image in q_images.items():
                if qname == q_image[1].name:
                    qcamera = q_cameras[q_image[1].camera_id]
                    pcr_images[qname] = [q_image[1].image_id, qcamera, tvec, qvec]
                    break
            
    # refine the pose of given image
    print('Starting pose refinement')
    refined_poses = []
    refine_options = pycolmap.AbsolutePoseRefinementOptions()
    refine_options.max_num_iterations = 500
    refine_options.refine_extra_params = False
    refine_options.refine_focal_length = False

    for pcr_img in tqdm(pcr_images):
        kpq = io.get_keypoints(features_path, pcr_img)
        kpq += 0.5  # COLMAP coordinates
        kp_idx_to_3D = defaultdict(list)
        kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
        db_names = parsers.parse_retrieval(retrievel_path)
        num_matches = 0
        db_ids = []
        db_imgs = db_names[pcr_img]
        for db_img in db_imgs:
            if db_img not in db_name_to_id:
                continue
            db_ids.append(db_name_to_id[db_img])
        for i, db_id in enumerate(db_ids):
            image = reference_sfm.images[db_id]
            if image.num_points3D() == 0:
                continue
            points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                    for p in image.points2D])

            matches, _ = io.get_matches(matches_path, pcr_img, image.name)
            matches = matches[:-1][points3D_ids[matches[:-1, 1]] != -1] # some indexing errors of the last element in the matches 
            num_matches += len(matches)
            for idx, m in matches:
                id_3D = points3D_ids[m]
                kp_idx_to_3D_to_db[idx][id_3D].append(i)
                # avoid duplicate observations
                if id_3D not in kp_idx_to_3D[idx]:
                    kp_idx_to_3D[idx].append(id_3D)
        
        idxs = list(kp_idx_to_3D.keys())
        mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
        mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
        msk = np.asarray(mkp_idxs) < len(kpq)
        mkp_idxs = np.asarray(mkp_idxs)[msk]
        mp3d_ids = np.asarray(mp3d_ids)[msk]
        
        points2D = kpq[mkp_idxs]
        points3D = [reference_sfm.points3D[j].xyz for j in mp3d_ids]
        colmap_img = query_sfm.images[pcr_images[qname][0]]#######
        colmap_img.tvec = pcr_images[pcr_img][2]
        colmap_img.qvec = pcr_images[pcr_img][3]
        colmap_cam = query_sfm.cameras[colmap_img.camera_id]
        
        # generate inlier mask based on pcr poses
        # project 3D points into image plane and check the projected 2d points with points2D
        projected_points2D = colmap_cam.world_to_image(colmap_img.project(points3D))
        rep_errs = np.linalg.norm(projected_points2D - points2D, axis=1)
        inlier_mask = rep_errs < 5 ## parameter needs tuning
        for i,point_2D in enumerate(points2D):
            ids = np.where((points2D == point_2D).all(axis=1))
            if rep_errs[i] == np.min(rep_errs[ids]) and rep_errs[i] < 15:
                inlier_mask[i] = True
            else:
                inlier_mask[i] = False
        refined_ret = pycolmap.pose_refinement(pcr_images[pcr_img][2], 
                                            pcr_images[pcr_img][3],
                                            points2D,
                                            points3D,
                                            inlier_mask,
                                            colmap_cam,
                                            refine_options
                                            )
        refined_corr = - read_write_model.qvec2rotmat(refined_ret['qvec']).T @ refined_ret['tvec']
        pcr_corr = - read_write_model.qvec2rotmat(pcr_images[pcr_img][3]).T @ pcr_images[pcr_img][2]
        if np.linalg.norm(refined_corr - pcr_corr) < 0.5:
            refined_poses.append([pcr_img, refined_ret['qvec'], refined_ret['tvec']])
        else:
            refined_poses.append([pcr_img, pcr_images[pcr_img][3], pcr_images[pcr_img][2]])
        
    with open(output, 'w') as f:
        for pose in refined_poses:
            qvec = ' '.join(map(str, pose[1]))
            coor = ' '.join(map(str, pose[2]))
            f.write(f'{pose[0]} {qvec} {coor}\n')
    
    return refined_poses
    
def rig_pose_refiner(
        pcr_results,
        features_path,
        matches_path,
        retrievel_path,
        query_sfm,
        reference_sfm, 
        output
    ):
    """refine the poses in the sequence jointly
    by calling pycolmap.rig_absolute_pose_estimation()
    codebase: hloc/localize_sfm.py
    Input args:
        pcr_results,
        q_cameras, 
        q_images,
        features_path,
        matches_path
    Output args:
        Camera_poses:[qname, qvec, tvec] of each camera

    """
    pcr_images = {}
    reference_sfm = pycolmap.Reconstruction(reference_sfm)
    query_sfm = pycolmap.Reconstruction(query_sfm)
    q_images = query_sfm.images
    q_cameras = query_sfm.cameras
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}
    with open(pcr_results, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            image = p.split()
            qname = image[0]
            qcamera = None
            qvec = np.asarray(image[1:5], dtype=np.float64)
            tvec = np.asarray(image[5:], dtype=np.float64)
            for q_image in q_images.items():
                if qname == q_image[1].name:
                    qcamera = q_cameras[q_image[1].camera_id]
                    pcr_images[qname] = [q_image[1].image_id, qcamera, tvec, qvec]
                    break
            
    # refine the pose of given image
    print('Starting rig pose refinement')
    refine_options = pycolmap.AbsolutePoseRefinementOptions()
    refine_options.max_num_iterations = 500
    refine_options.refine_extra_params = True
    refine_options.refine_focal_length = True

    points2D_all, points3D_all, cameras_all, rig_tvecs, rig_qvecs = [],[],[],[],[]
    for pcr_img in tqdm(pcr_images):
        kpq = io.get_keypoints(features_path, pcr_img)
        kpq += 0.5  # COLMAP coordinates
        kp_idx_to_3D = defaultdict(list)
        kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
        db_names = parsers.parse_retrieval(retrievel_path)
        num_matches = 0
        db_ids = []
        db_imgs = db_names[pcr_img]
        for db_img in db_imgs:
            if db_img not in db_name_to_id:
                continue
            db_ids.append(db_name_to_id[db_img])
        for i, db_id in enumerate(db_ids):
            image = reference_sfm.images[db_id]
            if image.num_points3D() == 0:
                continue
            points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                    for p in image.points2D])

            matches, _ = io.get_matches(matches_path, pcr_img, image.name)
            matches = matches[:-1][points3D_ids[matches[:-1, 1]] != -1] # some indexing errors of the last element in the matches 
            num_matches += len(matches)
            for idx, m in matches:
                id_3D = points3D_ids[m]
                kp_idx_to_3D_to_db[idx][id_3D].append(i)
                # avoid duplicate observations
                if id_3D not in kp_idx_to_3D[idx]:
                    kp_idx_to_3D[idx].append(id_3D)
        
        idxs = list(kp_idx_to_3D.keys())
        mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
        mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
        msk = np.asarray(mkp_idxs) < len(kpq)
        mkp_idxs = np.asarray(mkp_idxs)[msk]
        mp3d_ids = np.asarray(mp3d_ids)[msk]
        
        points2D = kpq[mkp_idxs]
        points3D = [reference_sfm.points3D[j].xyz for j in mp3d_ids]
        colmap_img = query_sfm.images[pcr_images[qname][0]]
        colmap_cam = query_sfm.cameras[colmap_img.camera_id]
        points2D_all.append(points2D)
        points3D_all.append(points3D)
        cameras_all.append(colmap_cam)
        rig_tvecs.append(pcr_images[pcr_img][2])
        rig_qvecs.append(pcr_images[pcr_img][3])
    
    rig_coord = [- read_write_model.qvec2rotmat(rig_qvecs[i]).T @ rig_tvecs[i] for i in range(0, len(rig_qvecs))]
    rig_coord -= np.mean(rig_coord, axis=0)
    rig_tvecs = [- read_write_model.qvec2rotmat(rig_qvecs[i]) @ rig_coord[i] for i in range(0, len(rig_qvecs))]
        
    ret = pycolmap.seq_absolute_pose_refiner(points2D_all, points3D_all, cameras_all, rig_qvecs, rig_tvecs, refinement_options = refine_options)
    
    # refine the camera poses based on refinement result
    rig_center = -read_write_model.qvec2rotmat(ret['qvec']).T @ ret['tvec']
    T_rig_to_world = np.eye(4)
    T_rig_to_world[0:3, 0:3] = read_write_model.qvec2rotmat(ret['qvec']).T
    T_rig_to_world[:3, 3] = rig_center
    refined_camera_poses = []
    rig_qvecs = ret['seq_qvecs']
    rig_tvecs = ret['seq_tvecs']
    for i, pcr_img in enumerate(pcr_images):
        # tvec, qvec -> rig frame
        rig_coord = -read_write_model.qvec2rotmat(rig_qvecs[i]).T @ rig_tvecs[i]
        rig_rot = read_write_model.qvec2rotmat(rig_qvecs[i]).T
        rig_projection_center = np.append(rig_rot, rig_coord.reshape((3,1)), axis=1)
        rig_projection_mtx = np.append(rig_projection_center, np.array([0,0,0,1]).reshape(1,4), axis = 0)
        world_projection_mtx = T_rig_to_world @ rig_projection_mtx
        world_rotmtx = world_projection_mtx[:3,:3]
        qvec = read_write_model.rotmat2qvec(world_rotmtx.T)
        tvec = - world_rotmtx.T @ world_projection_mtx[0:3,3]
        # refined_camera_poses.append([pcr_img, qvec, tvec])
        
        refined_coor = - read_write_model.qvec2rotmat(qvec).T @ tvec
        pcr_coor = - read_write_model.qvec2rotmat(pcr_images[pcr_img][3]).T @ pcr_images[pcr_img][2]
        if np.linalg.norm(refined_coor - pcr_coor) < 0.5:
            refined_camera_poses.append([pcr_img, qvec, tvec])
        else:
            refined_camera_poses.append([pcr_img, pcr_images[pcr_img][3], pcr_images[pcr_img][2]])
    
    if not output.exists(): output.touch()
    with open(output, 'w') as f:
        for pose in refined_camera_poses:
            qvec = ' '.join(map(str, pose[1]))
            coor = ' '.join(map(str, pose[2]))
            f.write(f'{pose[0]} {qvec} {coor}\n')

    return refined_camera_poses  


def main(db_model,query_model,local_visual:bool):
    ##  Path define
    if isinstance(db_model, str):
        db_model = Path(db_model)
    if isinstance(query_model, str):
        query_model = Path(query_model)

    pairs_3d_path = query_model/'3d_pairs.txt'
    sfm_model = query_model/'outputs'/"sfm_superpoint+superglue/"

    pairs_3d, scores = parse_3d_pairs(pairs_3d_path)
    _, q_images, q_points3D = read_write_model.read_model(path=sfm_model, ext='.bin')
    # _, _, ref_points3D = read_write_model.read_model(path=db_model / "sfm_superpoint+superglue/", ext='.bin')
    ref_points3D = read_write_model.read_points3D_binary(db_model / "sfm_superpoint+superglue/points3D.bin")
    corr_3d_path = query_model / "3d_corr.txt"
    write_3dpts_corr(pairs_3d, ref_points3D, q_points3D, corr_3d_path)
    query_pcd = query_model/'outputs'/"point_cloud.ply"
    ref_pcd = db_model/"point_cloud.ply"
    T, t_scale = teaser_pcr.main(corr_3d_path, str(query_pcd), str(ref_pcd), VISUALIZE=local_visual)
    localizer(q_images, T, t_scale,  query_model/'pcr_results.txt')
    pose_refiner(query_model/'pcr_results.txt', 
                 query_model/'outputs/feats-superpoint-n4096-r1024.h5', 
                 query_model/'qd_match.h5', 
                 query_model/'qd_pairs.txt',
                 sfm_model,
                 db_model/'sfm_superpoint+superglue',
                 query_model/'refined_results.txt'
                 )
    rig_pose_refiner(query_model/'refined_results.txt', 
                    query_model/'outputs/feats-superpoint-n4096-r1024.h5', 
                    query_model/'qd_match.h5', 
                    query_model/'qd_pairs.txt',
                    sfm_model,
                    db_model/'sfm_superpoint+superglue',
                    query_model/'rig_refined_results.txt'
                    )  
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate camera poses by point cloud registration')
    parser.add_argument('--db_model', type=str, required=True, help="Path to the database model")
    parser.add_argument('--query_model', type=str, required=True, help="Path to the query model")
    parser.add_argument('--local_visual', type=bool, default=False, help="Use open3d to visualize results")

    args = parser.parse_args()

    main(**args.__dict__)
    
    
    