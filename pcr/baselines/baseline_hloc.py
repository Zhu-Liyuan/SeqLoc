"""
Baseline - hloc to localize single image
"""
from pathlib import Path

from hloc import extract_features, match_features, pairs_from_retrieval
from hloc import localize_sfm

from pcr.utils import evaluate_results,save_eva_results
import argparse

def main(dataset_path,
         query_images_path,
         global_map_path,
         evaluation = True
         ):
    """
    Baseline: Use hloc to localize single image
    Args:
        dataset_path: Path to Aachen Day and Night
        query_images_path: Path to the image sequence to be localized
        global_map_path: Path to the pre-built Aachen global map
    """
    # Paths
    # Aachen dataset
    dataset = dataset_path

    # Local image sequence
    query_dir = query_images_path
    query_output = query_dir / 'outputs'
    query_images = query_dir / 'images' # Query images stored
    loc_pairs = query_output / 'pairs-top20-query-netvlad20.txt'  # top 20 retrieved by NetVLAD (global descriptors of query images)
    localization_results = query_dir / 'hloc_localization_results.txt'
    evaluation_result_path = query_dir / 'hloc_localization_evaluation.csv'

    # Global map
    global_dir = global_map_path
    reference_sfm = global_dir / 'sfm_superpoint+superglue'  # the reference SfM model we built
    reference_features = global_dir / 'feats-superpoint-n4096-r1024.h5' # reference superpoint features
    db_descriptors = global_dir/'global-feats-netvlad.h5'



    # pick one of the configurations for image retrieval, local feature extraction, and matching
    # you can also simply write your own here!
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    # Global descriptors by Netvlad
    global_descriptors = extract_features.main(retrieval_conf, query_images, query_output)
    pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=20,
                              db_descriptors=db_descriptors,
                              db_model=reference_sfm
                             )

    # Local features
    features = extract_features.main(feature_conf, query_images, query_output)

    # Match query images
    loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], query_output, features_ref=reference_features)

    # Localization
    localize_sfm.main(
        reference_sfm,
        dataset / '3D-models/database_intrinsics.txt',
        loc_pairs,
        features,
        loc_matches,
        localization_results,
        covisibility_clustering=False)  # not required with SuperPoint+SuperGlue

    # Evaluate
    if evaluation:
        gt = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_exp/ref/outputs/sfm_superpoint+superglue')
        eval_results = evaluate_results(gt / 'images.bin', localization_results)
        save_eva_results(eval_results, fpath=evaluation_result_path)


if __name__ == "__main__":
    dataset = Path('/home/marvin/ETH_Study/3DV/3DV/datasets/aachen')
    query_dir = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_sub2/query_sequences_v2')
    global_dir = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/new_triangulation/outputs')

    for i in range(11,12):
        main(dataset, query_dir/str(i), global_dir, evaluation=True)