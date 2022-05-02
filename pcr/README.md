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