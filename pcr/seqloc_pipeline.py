from pathlib import Path
import sfm_pipeline, pairs_3d_from_2d, poses_from_pcr
from utils import evaluate_results
import numpy as np
if __name__ == '__main__':
    query_path = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_exp/query')
    ref_path = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_exp/ref/')
    # query_path = Path('/home/marvin/ETH_Study/3DV/3DV/datasets/pcr/query/')
    # ref_path = Path('/home/marvin/ETH_Study/3DV/3DV/datasets/pcr/db')
    # sfm_pipeline.main(query_path, "netvlad", "superpoint_aachen", "superglue")
    # pairs_3d_from_2d.main(ref_path/'outputs', query_path/'outputs')
    # poses_from_pcr.main(ref_path/'outputs', query_path/'outputs')
    # evaluate_results(ref_path/'outputs/sfm_superpoint+superglue/images.bin', query_path/'localization_results.txt')
    
    results = np.empty((0,2))
    for i in range(1,11):
        query_str = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_sub/query/' + str(i))/'localization_results.txt'
        # results.append(evaluate_results(ref_path/'outputs/sfm_superpoint+superglue/images.bin', query_str))
        results = np.append(results,evaluate_results(ref_path/'outputs/sfm_superpoint+superglue/images.bin', query_str))
    
    results = np.array(results).reshape(-1,2)
    pass
