import numpy as np
from pathlib import Path
from hloc.utils import read_write_model, database, viz, io, parsers
from hloc import pairs_from_retrieval, match_features


def main():
    ##  Path define
    data_path = Path("/home/liyzhu/ETHZ/3DV/datasets")
    db_model = data_path/"db/outputs"
    query_model = data_path/"query/outputs"

    db_global_desc = db_model/"global-feats-netvlad.h5"
    query_global_desc = query_model/"global-feats-netvlad.h5"

    db_feat_desc = db_model/"feats-superpoint-n4096-r1024.h5"
    query_feat_desc = query_model/"feats-superpoint-n4096-r1024.h5"

    # db_db = database.COLMAPDatabase.connect(db_model/"database.db")
    # query_db = database.COLMAPDatabase.connect(query_model/"database.db")
    output = data_path/"test_pcr"


    ##  Generate image matches, 5 match per query image
    pairs_from_retrieval.main(query_global_desc, output/"qd_pairs.txt", num_matched = 5, db_descriptors=db_global_desc)

    ##  Point Feature matching between q and db
    conf = match_features.confs['superglue-fast']
    match_features.match_from_paths(conf = conf,
        pairs_path= output/"qd_pairs.txt", 
        match_path= output/"qd_match.h5", 
        feature_path_q= query_feat_desc, 
        feature_paths_refs= [db_feat_desc])

    ## 3D-2D correspondences
    _, images, points3D = read_write_model.read_model(path=query_model/"sfm_superpoint+superglue/", ext='.bin')

    pairs = parsers.parse_retrieval(output/"qd_pairs.txt")
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    hists = []
    # the orders of key_points indices in images.bin and feat.h5 are the same, indices in bin are float(~0.5) while indices in h5 are integer 
    for i in images:
        # print(i, images[i].point3D_ids)
        name = images[i].name
        xys = images[i].xys
        recon_mask = (images[i].point3D_ids!=-1) #mask of 2d points that are successfully reconstructed
        recon_ids = np.arange(len(xys))[recon_mask] # indices same to above
        recon_2d_pts = xys[recon_ids]
        
        hist = np.zeros(xys.shape[0])
        for q,r in pairs:
            if q == name:
                # kpts = io.get_keypoints(query_feat_desc, q)
                matches = io.get_matches(output/"qd_match.h5", q, r)
                qd_ids = matches[0][:,0] #indices of query_ref matchs in query
                intx = np.intersect1d(recon_ids, qd_ids) #find the indices that are reconstructed and matched in the image
                hist[intx]+=1
        hists.append(hist)
    

    print()





if __name__ == "__main__":
    main()
    
