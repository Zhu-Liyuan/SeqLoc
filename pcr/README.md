# Generate point cloud from image sequence & conduct point cloud registration

## Pipeline for **_seqloc_**

You can run the _seqloc_ pipeline **step by step** by using the following commands:

1. SFM to reconstruct local point cloud for image sequences: 

`python -m pcr.sfm_pipeline --proj_dir="/cluster/project/infk/courses/252-0579-00L/group16/PCR/AachenImageSequenceSamples/1"`

2. Extract 2d-to-3d correspondence: 

`python -m pcr.pairs_3d_from_2d --db_model="/cluster/project/infk/courses/252-0579-00L/group16/output/superpoint+superglue_aachen" --query_model="/cluster/project/infk/courses/252-0579-00L/group16/PCR/AachenImageSequenceSamples/1"`

3. Register local point cloud to the global model: 

`python -m pcr.poses_from_pcr --db_model="/cluster/project/infk/courses/252-0579-00L/group16/output/superpoint+superglue_aachen" --query_model="/cluster/project/infk/courses/252-0579-00L/group16/PCR/AachenImageSequenceSamples/1"`

Or you can run the _seqloc_ pipeline **as a whole** by calling:

`python -m pcr.seqloc_pipeline --config="pcr/config/seqloc.yaml"`

## Evaluation

### Baselines:

**1. Naive baseline:**

Given a image sequence, localize single image and calculate the localization error.

`python -m pcr.baselines.baseline_hloc`





## Data Structure for query and reference

```Shell
├── data
    ├── Outputs
        ├── database.db
        ├── point_feat_*.h5(point features)
        ├── matches_*.h5(feature matches)
        ├── flobal_feats_*.h5
        ├── pairs_netlad.txt
        ├── point_cloud.ply
        ├── sfm
            ├── points3D.bin
            ├── cameras.bin
            ├── images.bin
            ├── database.bin
            ├── models
    ├── images
        ├── Image1.JPG
        ├── Image2.JPG
        .
        .
        .
        ├── ImageX.JPG
        
```
The discription of image pose in .bin file goes here: https://colmap.github.io/format.html#sparse-reconstruction