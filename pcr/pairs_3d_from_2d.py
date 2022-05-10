import numpy as np
from pathlib import Path
from hloc.utils import read_write_model, database, viz, io, parsers
from hloc import pairs_from_retrieval, match_features
import h5py
from tqdm import tqdm
import argparse


def image_from_name(images, name):
    for item in images:
        if images[item].name == name:
            return images[item]


def main(db_model, query_model, output_dir):
    ##  Path define
    if isinstance(db_model, str):
        db_model = Path(db_model)
    if isinstance(query_model, str):
        query_model = Path(query_model)
    query_model = query_model / "outputs"

    db_global_desc = db_model / "global-feats-netvlad.h5"
    query_global_desc = query_model / "global-feats-netvlad.h5"

    db_feat_desc = db_model / "feats-superpoint-n4096-r1024.h5"
    query_feat_desc = query_model / "feats-superpoint-n4096-r1024.h5"

    # output = query_model / ".."
    # TODO output_dir
    output = Path(output_dir)
    if not output.exists():
        output.mkdir()
    qd_matches = output / "qd_match.h5"

    assert db_feat_desc.exists() * db_global_desc.exists() * query_feat_desc.exists() * query_global_desc.exists(), "Some feature files are not found!"

    ##  Generate image matches, 5 match per query image
    pairs_from_retrieval.main(query_global_desc, output / "qd_pairs.txt", num_matched=10, db_descriptors=db_global_desc)

    ##  Point Feature matching between q and db
    conf = match_features.confs['superglue']
    match_features.match_from_paths(conf=conf,
                                    pairs_path=output / "qd_pairs.txt",
                                    match_path=output / "qd_match.h5",
                                    feature_path_q=query_feat_desc,
                                    feature_paths_refs=[db_feat_desc])

    ## 3D-2D correspondences
    _, q_images, q_points3D = read_write_model.read_model(path=query_model / "sfm_superpoint+superglue/", ext='.bin')
    # ref_cameras, ref_images, ref_points3D = read_write_model.read_model(path=db_model / "sfm_superpoint+superglue/", ext='.bin')
    ref_images = read_write_model.read_images_binary(db_model / "sfm_superpoint+superglue/images.bin")
    pairs = parsers.parse_retrieval(output / "qd_pairs.txt")
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    hists = []
    # the orders of key_points indices in images.bin and feat.h5 are the same, indices in bin are (~0.5) while indices in h5 are integer 
    for i in q_images:
        # print(i, images[i].point3D_ids)
        name = q_images[i].name
        xys = q_images[i].xys
        recon_mask = (q_images[i].point3D_ids != -1)  # mask of 2d points that are successfully reconstructed
        recon_ids = np.arange(len(xys))[recon_mask]  # indices same to above
        recon_2d_pts = xys[recon_ids]

        hist = np.zeros(xys.shape[0])
        for q, r in pairs:
            if q == name:
                # kpts = io.get_keypoints(query_feat_desc, q)
                matches = io.get_matches(qd_matches, q, r)
                qd_ids = matches[0][:, 0]  # indices of query_ref matchs in query
                intx = np.intersect1d(recon_ids,
                                      qd_ids)  # find the indices that are reconstructed and matched in the image
                hist[intx] += 1
        hists.append(hist)

    pairs_3d = []
    NUM_2D_PTS = min(4, len(q_images) * 0.5)  # parameter that needs experiment
    # hfile_qr_matches = h5py.File(str(qd_matches), 'r')
    hfile_ref_kpts = h5py.File(str(db_feat_desc), 'r')
    # generate ref_image name list for fast indexing
    # ref_names = [ref_images[i].name for i in ref_images]
    # TO DO: rewrite find_matches, get_keypoints into class
    print('Generate 3d point pairs:')
    for q_id in tqdm(q_points3D):
        image_ids = q_points3D[q_id].image_ids  # num of corr 2d points for a 3d point
        q_pt2d_ids = q_points3D[q_id].point2D_idxs
        corr_3ds = np.array([], dtype=np.int64)
        if len(image_ids) > NUM_2D_PTS:
            for img_id, pt_id in zip(image_ids, q_pt2d_ids):
                q_img = q_images[img_id].name
                if q_images[img_id].point3D_ids[pt_id] == -1: continue
                for q, r in pairs:
                    if q == q_img:
                        matches = io.get_matches(qd_matches, q, r)[0]
                        ref_id = matches[matches[:, 0] == pt_id, 1]
                        # ref_image = ref_images[ref_names.index(r)] ## there might be bugs of images indexing
                        ref_image = image_from_name(ref_images, r)
                        # corr_3ds.append(ref_image.point3D_ids[ref_id])
                        if ref_id == -1 or len(ref_id) < 1 or ref_image == None:
                            continue
                        corr_3ds = np.append(corr_3ds, np.array(ref_image.point3D_ids[ref_id]))

        if corr_3ds.size != 0:
            uni_ids, uni_counts = np.unique(corr_3ds, return_counts=True)
            post_prob = max(uni_counts) / sum(uni_counts)
            if post_prob > 0.6:
                ref_3d = uni_ids[uni_counts == max(uni_counts)]
                pairs_3d.append((q_id, int(ref_3d), post_prob))

    # print(pairs_3d)
    with open(output / "3d_pairs.txt", 'w') as f:
        f.write('\n'.join(' '.join([str(pair[0]), str(pair[1]), str(pair[2])]) for pair in pairs_3d))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pairs from 3D to 2D')
    # TODO
    # parser.add_argument('--db_model', type=str, required=True, help="Path to the database model")
    # parser.add_argument('--query_model', type=str, required=True, help="Path to the query model")
    parser.add_argument('--db_model', type=str,default="/cluster/project/infk/courses/252-0579-00L/group16/output"
                                                       "/superpoint+superglue_aachen")
    parser.add_argument('--query_model', type=str, default="/cluster/project/infk/courses/252-0579-00L/group16/hs"
                                                           "/test/sfm_pipeline")
    parser.add_argument('--output_dir', type=str,
                        default='/cluster/project/infk/courses/252-0579-00L/group16/hs/test/pairs_3d_from_2d')

    args = parser.parse_args()

    main(**args.__dict__)
