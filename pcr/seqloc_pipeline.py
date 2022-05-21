from pathlib import Path
from pcr import sfm_pipeline, pairs_3d_from_2d, poses_from_pcr
from pcr.utils import load_config, evaluate_results, save_eva_results

import argparse


def main(config_path:str):
    """
    sequential localization pipeline
    Args:
        config_path: Path to config file
    """
    # Load config file
    config = load_config(config_path)
    # Define paths
    ref_path = Path(config["database"]["proj_dir"])

    # TODO
    for i in range(2, 20):
        query_path = Path(config["query"]["proj_dir"])
        # TODO
        query_path = query_path / str(i)
        print(query_path)
        # Seqloc pipeline
        # TODO
        sfm_pipeline.main(query_path, **config["query_params"])
        pairs_3d_from_2d.main(ref_path, query_path)
        poses_from_pcr.main(ref_path, query_path, local_visual=config['query']['local_visual'])
        # Evaluation
        # TODO
        eval_path = Path('/cluster/project/infk/courses/252-0579-00L/group16/output/superpoint+superglue_aachen')
        eval_results = evaluate_results(eval_path/'sfm_superpoint+superglue/images.bin', query_path/'refined_results.txt')
        save_eva_results(eval_results, query_path / "seqloc_evaluation.csv")

        eval_results = evaluate_results(eval_path/'sfm_superpoint+superglue/images.bin', query_path/'rig_refined_results.txt')
        save_eva_results(eval_results, query_path / "rig_seqloc_evaluation.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seqloc pipeline')
    parser.add_argument('--config_path', type=str, default="config/seqloc.yaml", help="Path to the config file")

    args = parser.parse_args()

    main(**args.__dict__)




