from pathlib import Path
from pcr import sfm_pipeline, pairs_3d_from_2d, poses_from_pcr
from pcr.utils import load_config, evaluate_results
import numpy as np

import argparse


def main(config_path):
    # Load config file
    config = load_config(config_path)
    # Define paths
    ref_path = Path(config["database"]["proj_dir"])
    query_path = Path(config["query"]["proj_dir"])
    # Seqloc pipeline
    sfm_pipeline.main(**config["query"])
    pairs_3d_from_2d.main(ref_path, query_path)
    poses_from_pcr.main(ref_path, query_path)
    # Evaluation
    evaluate_results(ref_path/'outputs/sfm_superpoint+superglue/images.bin', query_path/'localization_results.txt')

    # results = np.empty((0, 2))
    # for i in range(1, 11):
    #     query_str = Path(
    #         '/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_sub/query/' + str(i)) / 'localization_results.txt'
    #     # results.append(evaluate_results(ref_path/'outputs/sfm_superpoint+superglue/images.bin', query_str))
    #     results = np.append(results,
    #                         evaluate_results(ref_path / 'outputs/sfm_superpoint+superglue/images.bin', query_str))
    #
    # results = np.array(results).reshape(-1, 2)
    # pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seqloc pipeline')
    parser.add_argument('--config_path', type=str, required=True, help="Path to the config file")

    args = parser.parse_args()

    main(**args.__dict__)




