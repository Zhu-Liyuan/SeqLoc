from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
import argparse


def main(proj_dir,
         output_dir,
         retrieval_conf,
         feature_conf,
         matcher_conf, local_visual):
    """
    SFM pipeline
    Input: Downloaded images from Aachen Day & Night in `img_dir`
    Output: sfm features & reconstructed model in `outputs`
    """
    img_dir = Path(proj_dir) / 'images'
    # TODO
    # outputs = Path(proj_dir) / 'outputs/'
    outputs = Path(output_dir) / 'outputs/'
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    sfm_dir = outputs / 'sfm_superpoint+superglue'

    retrieval_conf = extract_features.confs[retrieval_conf]
    feature_conf = extract_features.confs[feature_conf]
    matcher_conf = match_features.confs[matcher_conf]
    # Find image pairs via image retrieval
    retrieval_path = extract_features.main(retrieval_conf, img_dir, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)
    # Extract features for image pairs; and match local features
    feature_path = extract_features.main(feature_conf, img_dir, outputs)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
    # 3D Reconstruction
    model = reconstruction.main(sfm_dir, img_dir, sfm_pairs, feature_path, match_path)

    if local_visual:
        from pcr.utils import convert_bin_to_ply
        convert_bin_to_ply(sfm_dir, sfm_dir / '../point_cloud.ply')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SfM configurations')
    # parser.add_argument('--proj_dir', type=str, required=True)
    # TODO
    parser.add_argument('--proj_dir', type=str,
                        default='/cluster/project/infk/courses/252-0579-00L/group16/PCR/query_sequencies/1')
    parser.add_argument('--output_dir', type=str,
                        default='/cluster/project/infk/courses/252-0579-00L/group16/hs/test/sfm_pipeline')
    parser.add_argument('--retrieval_conf', type=str, default="netvlad", choices=list(extract_features.confs.keys()))
    parser.add_argument('--feature_conf', type=str, default="superpoint_aachen",
                        choices=list(extract_features.confs.keys()))
    parser.add_argument('--matcher_conf', type=str, default="superglue", choices=list(match_features.confs.keys()))
    parser.add_argument('--local_visual', type=bool, default=False, help="Use open3d to visualize locally")

    args = parser.parse_args()

    main(**args.__dict__)
