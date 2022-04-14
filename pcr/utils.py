from pathlib import Path
import numpy as np
from hloc.utils import read_write_model
import open3d as o3d
import yaml
from yaml import CLoader as Loader, CDumper as Dumper

def convert_bin_to_ply(input_path, output_path):
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


if __name__ == "__main__":
    input_path = Path("/home/marvin/ETH_Study/3DV/3DV/datasets/pcr/db/outputs/sfm_superpoint+superglue")
    output_path = Path("/home/marvin/ETH_Study/3DV/3DV/datasets/pcr/db/outputs/point_cloud.ply")
    # convert_bin_to_ply(input_path, output_path)
    config_path = Path("/home/marvin/ETH_Study/3DV/3DV/pcr/config/config_demo.yaml")
    load_config(str(config_path))