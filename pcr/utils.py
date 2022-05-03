from pathlib import Path
import numpy as np
from hloc.utils import read_write_model, database, io
import yaml
from yaml import CLoader as Loader, CDumper as Dumper


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
    import pycolmap
    from hloc import triangulation
    proj = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_sub/ref/outputs')
    reference_path = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_exp/ref/outputs/sfm_sift/')
    sfm_path = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_sub/ref/outputs/sfm_sift/')
    new_database = proj/'sfm_superpoint+superglue/database.db'
    new_proj = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_sub/ref/')
    reference = pycolmap.Reconstruction(reference_path)
    image_ids = triangulation.create_db_from_model(reference, new_database, new_proj/'images')
    triangulation.import_features(image_ids, new_database, proj/'feats-superpoint-n4096-r1024.h5')
    triangulation.import_matches(image_ids, new_database, proj/'pairs-netvlad.txt', proj/'feats-superpoint-n4096-r1024_matches-superglue_pairs-netvlad.h5',
                   None, False)
    
    if not sfm_path.exists():
        sfm_path.mkdir(parents=True, exist_ok=True)
    
    images = {}
    open(sfm_path/'points3D.bin', "wb")
    db = database.COLMAPDatabase.connect(new_database)
    q_images = db.execute("SELECT * FROM images")
    # q_cameras = db.execute("SELECT * FROM cameras")
    #### generate bin files from  original data
    ref_cameras, ref_images, _ = read_write_model.read_model(path=reference_path, ext='.bin')
    image_ids = [q_image[0] for q_image in q_images]
    for _, ref_image in ref_images.items():
        if ref_image.id in image_ids:
            kpts = io.get_keypoints(proj/"feats-superpoint-n4096-r1024.h5", ref_image.name)
            images[ref_image.id] = read_write_model.Image(
                id=ref_image.id, qvec=ref_image.qvec, tvec=ref_image.tvec,
                camera_id=ref_image.camera_id, name=ref_image.name,
                xys=kpts, point3D_ids=[])
    
    read_write_model.write_images_text(images, sfm_path/'images.bin')
    read_write_model.write_cameras_text(ref_cameras, sfm_path/'cameras.bin')
    
    ### triangulate submodules
    sub_reference = pycolmap.Reconstruction(sfm_path)
    reconstruction = triangulation.run_triangulation(
        sfm_path/'../sfm_superpoint+superglue',
        proj/'sfm_superpoint+superglue/database.db',
        Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_sub/ref/images/'),
        sub_reference)


def get_image_from_name(ref_images, name):
    # ref_images = read_write_model.read_images_binary(ref)
    for ref_image in ref_images.items():
        ref_name = ref_image[1].name.split('/')[-1]
        if name == ref_name:
            return ref_image


def angle_between_two_qvec(qvec1, qvec2):
    qvec2[1:] *= -1
    z = np.dot(qvec1, qvec2)
    return np.arccos(z)


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
            ref_img = get_image_from_name(ref_images, name)[1]
            qvec_1 = np.asarray(data[:4],dtype=np.float64)
            qvec_2 = ref_img.qvec
            ang_diff = angle_between_two_qvec(qvec_1, qvec_2)
            diff_angle = (min(ang_diff,np.pi - ang_diff)*180/np.pi)
            diff_distance = np.linalg.norm(np.asarray(data[4:],dtype=np.float64) - ref_img.tvec)
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

    
if __name__ == "__main__":
    input_path = Path("/home/marvin/ETH_Study/3DV/3DV/datasets/pcr/db/outputs/sfm_superpoint+superglue")
    output_path = Path("/home/marvin/ETH_Study/3DV/3DV/datasets/pcr/db/outputs/point_cloud.ply")
    # convert_bin_to_ply(input_path, output_path)
    config_path = Path("/home/marvin/ETH_Study/3DV/3DV/pcr/config/config_demo.yaml")
    load_config(str(config_path))
    # triangulate_sub_model(1,1)
    ref = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_exp/ref/outputs/sfm_sift/images.bin')
    result = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_sub/query/1/localization_results.txt')
    evaluate_results(ref,result)