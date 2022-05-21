from pathlib import Path
from nbformat import read
import numpy as np
from hloc import triangulation
from hloc.utils import read_write_model, database, parsers
import yaml
from yaml import CLoader as Loader, CDumper as Dumper
from teaser_pcr import parse_3d_corr
import pycolmap

def convert_bin_to_ply(input_path, output_path):
    ''' Convert bin to ply
    Args:
    '''
    import open3d as o3d
    points = read_write_model.read_points3D_binary(input_path/'points3D.bin')
    xyz = []
    rgb = []
    for point in points:
        xyz.append(points[point].xyz)
        rgb.append(points[point].rgb / 255)

    xyz = np.asarray(xyz).reshape((-1,3))
    rgb = np.asarray(rgb).reshape((-1,3))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    o3d.io.write_point_cloud(str(output_path), pcd)


def load_config(path):
    ''' Loads config file.
    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=Loader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from', None)

    # Include main configuration
    # update_recursive(cfg, cfg_special)

    return cfg_special


def triangulate_sub_model(reference_path, output_path):
    #### Generate new database for the sub model
    
    from hloc import triangulation
    proj = Path('/home/liyzhu/ETHZ/3DV/datasets/aachen_sub')
    reference_path = Path('/home/liyzhu/ETHZ/3DV/outputs/superpoint+superglue_aachen/sfm_sift/')
    sfm_path = Path('/home/liyzhu/ETHZ/3DV/datasets/aachen_sub/sfm_superpoint+superglue/')
    new_database = proj/'sfm_superpoint+superglue/database.db'
    # new_proj = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_sub/ref/')
    # reference = pycolmap.Reconstruction(reference_path)
    # image_ids = triangulation.create_db_from_model(reference, new_database, proj/'images')
    # triangulation.import_features(image_ids, new_database, proj/'outputs/feats-superpoint-n4096-r1024.h5')
    # triangulation.import_matches(image_ids, new_database, proj/'outputs/pairs-db-covis20.txt', proj/'outputs/feats-superpoint-n4096-r1024_matches-superglue_pairs-db-covis20.h5',
    #                None, False)
    
    # if not sfm_path.exists():
    #     sfm_path.mkdir(parents=True, exist_ok=True)
    
    # images = {}
    # open(sfm_path/'points3D.bin', "wb")
    # db = database.COLMAPDatabase.connect(new_database)
    # q_images = db.execute("SELECT * FROM images")
    # # q_cameras = db.execute("SELECT * FROM cameras")
    # #### generate bin files from  original data
    # ref_cameras, ref_images, _ = read_write_model.read_model(path=reference_path, ext='.bin')
    # image_ids = [q_image[0] for q_image in q_images]
    # for _, ref_image in ref_images.items():
    #     if ref_image.id in image_ids:
    #         kpts = io.get_keypoints(proj/"outputs/feats-superpoint-n4096-r1024.h5", ref_image.name)
    #         images[ref_image.id] = read_write_model.Image(
    #             id=ref_image.id, qvec=ref_image.qvec, tvec=ref_image.tvec,
    #             camera_id=ref_image.camera_id, name=ref_image.name,
    #             xys=kpts, point3D_ids=np.full((kpts.shape[0]),-1))
    
    # read_write_model.write_images_text(images, sfm_path/'images.txt')
    # read_write_model.write_cameras_binary(ref_cameras, sfm_path/'cameras.bin')
    
    ### triangulate submodules
    sub_reference = pycolmap.Reconstruction(sfm_path)
    reconstruction = triangulation.run_triangulation(
        sfm_path/'../sub_triagulate',
        proj/'sfm_superpoint+superglue/database.db',
        Path('/home/liyzhu/ETHZ/3DV/datasets/aachen_sub/images/'),
        sub_reference,
        verbose=True)


def get_image_from_name(ref_images, name):
    # ref_images = read_write_model.read_images_binary(ref)
    for ref_image in ref_images.items():
        ref_name = ref_image[1].name.split('/')[-1]
        if name == ref_name:
            return ref_image


def angle_between_two_qvec(qvec1, qvec2):
    # qvec2[1:] *= -1
    
    rmtx1 = read_write_model.qvec2rotmat(qvec1)
    rmtx2 = read_write_model.qvec2rotmat(qvec2)
    qvec = read_write_model.rotmat2qvec(rmtx1 @ rmtx2.T)
    ang = np.arccos(qvec[0])/2
    return ang


def evaluate_results(ref:Path, q_results:Path):
    """
    Args:
        ref: Reference path to *.bin
        q_results: Results path to localization results
    """
    assert ref.exists(),ref
    assert q_results.exists(),q_results
    ref_images = read_write_model.read_images_binary(ref)
    print(f'Run evaluation:(degree,meter)')
    results = []
    with open(q_results, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            name = name.split("/")[-1]
            ref_img = get_image_from_name(ref_images, name)[1]
            qvec_1 = np.asarray(data[:4],dtype=np.float64)
            qvec_2 = ref_img.qvec
            diff_angle = angle_between_two_qvec(qvec_1, qvec_2)*180/np.pi
            ref_coor = - read_write_model.qvec2rotmat(qvec_2).T @ ref_img.tvec
            query_coor = -read_write_model.qvec2rotmat(qvec_1).T @ np.asarray(data[4:],dtype=np.float64)
            diff_distance = np.linalg.norm(ref_coor - query_coor)
            # diff_distance = np.linalg.norm(np.asarray(data[4:],dtype=np.float64) - ref_img.tvec)
            print(f'{name}:({diff_angle},{diff_distance})')
            results.append([name, diff_angle, diff_distance])
    # results = np.asarray(results,dtype=np.float32).reshape(-1,2)
    return results


def save_eva_results(results_list: list, fpath: str):
    """
    Save evaluation results
    Args:
        results_list: [[img_name, diff_anlge_in_degree, diff_distance_in_meter],]
        fpath: save path
    """
    with open(fpath, "w") as file:
        file.writelines("image,angle,distance\n")
        for r in results_list:
            file.writelines(f"{r[0]},{r[1]},{r[2]}\n")

def visualize_cameras(localizations, sfm_model):
    """
    Plot camera pose estimate in ply file
    Args:
    
    """
    import open3d as o3d
    sfm_images = read_write_model.read_images_binary(sfm_model/'images.bin')
    sfm_cameras = read_write_model.read_cameras_binary(sfm_model/'cameras.bin')
    sfm_cam_vis, camera_vis = [], []
    with open(localizations, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            name = name.split("/")[-1]
            img = get_image_from_name(sfm_images, name)[1]
            cam = sfm_cameras[img.camera_id]
            intrinsics, extrinsics = np.eye(3), np.eye(4)
            intrinsics[0,0] = cam.params[0]
            intrinsics[1,1] = cam.params[0]
            intrinsics[0,2] = cam.params[1]
            intrinsics[1,2] = cam.params[2]
            qvec = np.asarray(data[:4],dtype=np.float64)
            tvec = np.asarray(data[4:],dtype=np.float64)
            coord = -read_write_model.qvec2rotmat(qvec).T @ tvec
            rmtx = read_write_model.qvec2rotmat(qvec).T
            extrinsics[0:3,0:3] = rmtx
            extrinsics[0:3,3] = coord
            vis = o3d.geometry.LineSet.create_camera_visualization(
                cam.width, 
                cam.height, 
                intrinsics, 
                extrinsics, 
                1.0)
            vis.paint_uniform_color([1.0, 0.5, 0.0])
            camera_vis.append(vis)
    for img in sfm_images.items():
        intrinsics, extrinsics = np.eye(3), np.eye(4)
        cam = sfm_cameras[img[1].camera_id]
        intrinsics[0,0] = cam.params[0]
        intrinsics[1,1] = cam.params[0]
        intrinsics[0,2] = cam.params[1]
        intrinsics[1,2] = cam.params[2]
        qvec = img[1].qvec
        tvec = img[1].tvec
        coord = -read_write_model.qvec2rotmat(qvec).T @ tvec
        rmtx = read_write_model.qvec2rotmat(qvec).T
        extrinsics[0:3,0:3] = rmtx
        extrinsics[0:3,3] = coord
        vis = o3d.geometry.LineSet.create_camera_visualization(
            cam.width, 
            cam.height, 
            intrinsics, 
            extrinsics, 
            1.0)
        vis.paint_uniform_color([0.0, 0.5, 1.0])
        sfm_cam_vis.append(vis)
    # o3d.visualization.draw_geometries(sfm_cam_vis)
    # o3d.visualization.draw_geometries(camera_vis)
    
    # o3d.io.write_line_set(str(localizations.parent/'sfm_cameras.ply'), sfm_cam_vis)
    # o3d.io.write_line_set(str(localizations.parent/'localized_cameras.ply'), camera_vis)
    return sfm_cam_vis, camera_vis
        
def visualize_all(ref_path, query_path):
    import open3d as o3d
    pcd_1 = query_path/'outputs/point_cloud.ply'
    pcd_2 = ref_path/'point_cloud.ply'
    localizations = query_path/'refined_results.txt'
    sfm_model = query_path/'outputs/sfm_superpoint+superglue'
    sfm_cam_vis, camera_vis = visualize_cameras(localizations, sfm_model)
    A_pcd = o3d.io.read_point_cloud(str(pcd_1))
    B_pcd = o3d.io.read_point_cloud(str(pcd_2))
    # A_pcd = A_pcd.voxel_down_sample(voxel_size=0.04)
    # B_pcd = B_pcd.voxel_down_sample(voxel_size=0.04)
    A_pcd.paint_uniform_color([1.0, 0.5, 0.0])
    # plot A and B 
    query_vis = [A_pcd]
    for line_set in sfm_cam_vis:
        query_vis.append(line_set)
    o3d.visualization.draw_geometries(query_vis)
    o3d.visualization.draw_geometries([B_pcd])

    #vis correspondences
    corr_file = query_path/'3d_corr.txt'
    A_corr, B_corr = parse_3d_corr(corr_file)
    num_corrs = A_corr.shape[1]
    points = np.concatenate((A_corr.T,B_corr.T),axis=0)
    lines = []
    for i in range(num_corrs):
        lines.append([i,i+num_corrs])
    colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([A_pcd,B_pcd,line_set])

    A_pcd_localized = o3d.io.read_point_cloud(str(query_path/'outputs/localized.ply'))
    # A_pcd_localized.scale(10, center=(0, 0, 0))
    # B_pcd.scale(10, center=(0, 0, 0))
    vis_localized = [A_pcd_localized,B_pcd]
    for cam in camera_vis:
        vis_localized.append(cam)
    o3d.visualization.draw_geometries(vis_localized)

def image_deleter(input_path, output_path, image_list):
    images = read_write_model.read_images_binary(input_path/'images.bin')
    reference_sfm = pycolmap.Reconstruction(input_path)
    image_names = parsers.parse_image_list(image_list)
    new_images = {i:images[i] for i in images if images[i].name not in image_names}
    read_write_model.write_images_binary(new_images, output_path/'images.bin')
    for i, image in images.items():
        if image.name in image_names:
            reference_sfm.deregister_image(i)
    
    reference_sfm.write_binary(str(output_path))

            
    


    
if __name__ == "__main__":
    camera_poses = Path("/home/marvin/ETH_Study/3DV/3DV/datasets/pcr/db/outputs/sfm_superpoint+superglue")
    ref_path = Path("/home/liyzhu/ETHZ/3DV/outputs/superpoint+superglue_aachen")
    query_path = Path('/home/liyzhu/ETHZ/3DV/outputs/query_sequencies/1')
    
    
    # load_config(str(config_path))
    # triangulate_sub_model(1,1)
    ref = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_exp/ref/outputs/sfm_sift/images.bin')
    result = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_sub/query/1/localization_results.txt')
    # evaluate_results(ref,result)

    # image_deleter(Path('/home/liyzhu/ETHZ/3DV/datasets/aachen_reference'),
                # Path('/home/liyzhu/ETHZ/3DV/datasets/aachen_sub/sfm_superpoint+superglue'),
                # Path('/home/liyzhu/ETHZ/3DV/datasets/aachen_sub/image_names_path.txt'))
    
    # visualize_cameras(Path('/home/liyzhu/ETHZ/3DV/outputs/query_sequencies/1/refined_results.txt',),
    #         Path("/home/liyzhu/ETHZ/3DV/outputs/query_sequencies/1/outputs/sfm_superpoint+superglue"))
        
    # visualize_all(ref_path, query_path)
    sfm_dir = Path('/home/liyzhu/ETHZ/3DV/datasets/aachen_sub/sub_triagulate')
    reference_model = Path('/home/liyzhu/ETHZ/3DV/datasets/aachen_sub/sfm_superpoint+superglue')
    image_dir = Path('/home/liyzhu/ETHZ/3DV/datasets/aachen_sub/images')
    pairs = Path('/home/liyzhu/ETHZ/3DV/datasets/aachen_sub/outputs/pairs-db-covis20.txt')
    features_path = Path('/home/liyzhu/ETHZ/3DV/datasets/aachen_sub/outputs/feats-superpoint-n4096-r1024.h5')
    match_path = Path('/home/liyzhu/ETHZ/3DV/datasets/aachen_sub/outputs/feats-superpoint-n4096-r1024_matches-superglue_pairs-db-covis20.h5')
    triangulation.main(sfm_dir, reference_model, image_dir, pairs, features_path, match_path, verbose=True)