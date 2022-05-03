"""
Baseline - hloc to localize single image
"""
from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization


def main():
    # Paths
    # Aachen dataset
    dataset = Path('/cluster/project/infk/courses/252-0579-00L/group16/Aachen-Day-Night')
    # Local image sequence
    query_dir = Path('/cluster/project/infk/courses/252-0579-00L/group16/PCR/AachenImageSequenceSamples') / '1'
    query_images = query_dir / 'images' # Query images stored
    loc_pairs = query_dir / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD (global descriptors of query images)
    results = query_dir / 'Aachen_hloc_superpoint+superglue_netvlad20.txt'
    # Global map
    global_dir = Path('/cluster/project/infk/courses/252-0579-00L/group16/output/superpoint+superglue_aachen')
    sfm_pairs = global_dir / 'pairs-db-covis20.txt'  # top 20 most covisible in SIFT model
    reference_sfm = global_dir / 'sfm_superpoint+superglue'  # the reference SfM model we built
    reference_features = global_dir / 'feats-superpoint-n4096-r1024.h5' # reference superpoint features
    # results = global_dir / 'Aachen_hloc_superpoint+superglue_netvlad20.txt'  # the result file


    # pick one of the configurations for image retrieval, local feature extraction, and matching
    # you can also simply write your own here!
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    # Global descriptors by Netvlad
    global_descriptors = extract_features.main(retrieval_conf, query_images, query_dir)
    pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=20,
                              db_descriptors=global_dir/'global-feats-netvlad.h5',
                              db_model=global_dir/'sfm_superpoint+superglue'
                             )

    # Local features
    features = extract_features.main(feature_conf, query_images, query_dir)

    # Match query images
    loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], query_dir, features_ref=reference_features)

    # Reformat loc_pairs (48.jpg to db/48.jpg)
    with open(loc_pairs, 'r') as file:
        data = file.read().split('\n')
    with open(query_dir / 'pairs-query-netvlad20-formated.txt', "w") as file:
        for d in data:
            images = d.split(" ")
            file.writelines(f"db/{images[0]} {images[1]}\n")
    loc_pairs = query_dir / 'pairs-query-netvlad20-formated.txt'

    # Localization
    localize_sfm.main(
        reference_sfm,
        dataset / '3D-models/database_intrinsics.txt',
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False)  # not required with SuperPoint+SuperGlue


if __name__ == "__main__":
    main()