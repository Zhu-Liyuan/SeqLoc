from pathlib import Path
import sfm_pipeline, pairs_3d_from_2d, poses_from_pcr


if __name__ == '__main__':
    query_path = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_sub/query/10')
    ref_path = Path('/home/marvin/ETH_Study/3DV/3DV/outputs/aachen_exp/ref/')
    sfm_pipeline.main(query_path, "netvlad", "superpoint_aachen", "superglue")
    pairs_3d_from_2d.main(ref_path/'outputs', query_path/'outputs')
    poses_from_pcr.main(ref_path/'outputs', query_path/'outputs')