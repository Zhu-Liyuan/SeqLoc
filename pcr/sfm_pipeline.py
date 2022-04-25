from pathlib import Path
from hloc import extract_features, match_features, reconstruction, pairs_from_retrieval
import argparse
from utils import convert_bin_to_ply
from hloc.utils import database, parsers

def main(proj_dir,
    retrieval_conf, 
    feature_conf,
    matcher_conf):
    
    img_dir = Path(proj_dir)/'images'
    outputs = Path(proj_dir) / 'outputs/'
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    sfm_dir = outputs / 'sfm_superpoint+superglue'
    intrinsics_dir = Path("/home/marvin/ETH_Study/3DV/3DV/datasets/aachen/3D-models/database_intrinsics.txt")
    # prior_poses_dir = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_exp/ref/outputs/sfm_sift')
    retrieval_conf = extract_features.confs[retrieval_conf]
    feature_conf = extract_features.confs[feature_conf]
    matcher_conf = match_features.confs[matcher_conf]
    img_loader = extract_features.ImageDataset(img_dir, feature_conf['preprocessing'])
    
    retrieval_path = extract_features.main(retrieval_conf, img_dir, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=len(img_loader) - 1)

    feature_path = extract_features.main(feature_conf, img_dir, outputs)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
    
    model = reconstruction.main(sfm_dir, img_dir, sfm_pairs, feature_path, match_path, intrinsics_path=intrinsics_dir)
    
    convert_bin_to_ply(sfm_dir, sfm_dir/'../point_cloud.ply')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SfM configurations')
    parser.add_argument('--proj_dir', type=str, default='/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_sub/query/3')
    parser.add_argument('--retrieval_conf', type=str, default="netvlad",choices=list(extract_features.confs.keys()))
    parser.add_argument('--feature_conf', type=str, default="superpoint_aachen", choices=list(extract_features.confs.keys()))
    parser.add_argument('--matcher_conf', type=str, default="superglue", choices=list(match_features.confs.keys()))
    
    
    
    args = parser.parse_args()

    main(**args.__dict__)
    