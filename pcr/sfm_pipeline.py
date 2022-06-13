from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
import argparse


def main(proj_dir,
    retrieval_conf, 
    feature_conf,
    matcher_conf, local_visual):
    """
    SFM pipeline
    Input: Downloaded images from Aachen Day & Night in `img_dir`
    Output: sfm features & reconstructed model in `outputs`
    """
    img_dir = Path(proj_dir)/'images'
    outputs = Path(proj_dir) / 'outputs/'
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    sfm_dir = outputs / 'sfm_superpoint+superglue'
    intrinsic_path = Path('/home/marvin/ETH_Study/3DV/3DV/datasets/aachen/3D-models/database_intrinsics.txt')
    retrieval_conf = extract_features.confs[retrieval_conf]
    feature_conf = extract_features.confs[feature_conf]
    matcher_conf = match_features.confs[matcher_conf]
    # Find image pairs via image retrieval
    img_loader = extract_features.ImageDataset(img_dir, feature_conf['preprocessing'])
    retrieval_path = extract_features.main(retrieval_conf, img_dir, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=len(img_loader) - 1)
    # Extract features for image pairs; and match local features
    feature_path = extract_features.main(feature_conf, img_dir, outputs)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
    # 3D Reconstruction
    model = reconstruction.main(sfm_dir, img_dir, sfm_pairs, feature_path, match_path, intrinsics_path=intrinsic_path)

    if local_visual:
        from pcr.utils import convert_bin_to_ply
        convert_bin_to_ply(sfm_dir, sfm_dir/'../point_cloud.ply')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SfM configurations')
    parser.add_argument('--proj_dir', type=str, required=True)
    parser.add_argument('--retrieval_conf', type=str, default="netvlad",choices=list(extract_features.confs.keys()))
    parser.add_argument('--feature_conf', type=str, default="superpoint_aachen", choices=list(extract_features.confs.keys()))
    parser.add_argument('--matcher_conf', type=str, default="superglue", choices=list(match_features.confs.keys()))
    parser.add_argument('--local_visual', type=bool, default=False, help="Use open3d to visualize locally")


    args = parser.parse_args()

    main(**args.__dict__)
    