from pathlib import Path
from pcr.utils import convert_bin_to_ply

if __name__ == "__main__":
    sfm_dir = Path('') / 'sfm_superpoint+superglue'
    convert_bin_to_ply(sfm_dir, sfm_dir/'../point_cloud.ply')