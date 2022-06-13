from hloc.utils import read_write_model, database, viz, io

if __name__ == "__main__":
    input_model = "/home/marvin/ETH_Study/3DV/3DV/datasets/pcr/db/"
    input_format = ".bin"
    cameras, images, points3D = read_write_model.read_model(path=input_model, ext=input_format)
    
    db = database.COLMAPDatabase.connect(input_model+"database.db")
    
    #vis.plot_matches
    image_name1 = "P1180156.JPG"
    image_name2 = "P1180157.JPG"
    p1 = io.get_keypoints(input_model+"outputs/feats-superpoint-n4096-r1024.h5", image_name1)
    p2 = io.get_keypoints(input_model+"outputs/feats-superpoint-n4096-r1024.h5", image_name2)
    matches = io.get_matches(input_model+"outputs/feats-superpoint-n4096-r1024_matches-superglue_pairs-netvlad.h5", image_name1, image_name2)
    img1 = io.read_image(input_model + image_name1)
    img2 = io.read_image(input_model + image_name2)
    viz.plot_images([img1,img2])
    kpts0 = p1[matches[0][:,0]]
    kpts1 = p2[matches[0][:,1]]
    viz.plot_matches(kpts0, kpts1, a=0.1)
    