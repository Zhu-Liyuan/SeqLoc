from pathlib import Path
import parser
import numpy as np

def parse_3d_pairs(path):
    pairs_3d = np.empty((0,3),dtype=np.int32)
    scores_3d = np.empty(0)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            if len(p) == 0:
                continue
            q, r, prob = p.split()
            pairs_3d = np.append(pairs_3d, np.asarray([q,r], dtype=np.int32))
            scores_3d = np.append(scores_3d, np.asarray([prob], dtype=np.float64))
    
    return pairs_3d.reshape(-1,2), scores_3d


def compute_scale():
    raise NotImplementedError()

def get_poses_from_corr():
    raise NotImplementedError()



if __name__ == '__main__':
    path = '/home/marvin/ETH_Study/3DV/3DV/datasets/pcr/q_ref_match/3d_pairs.txt'
    parse_3d_pairs(path)