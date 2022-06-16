# SeqLoc: Visual Localization with an Image Sequence
### ****Student Project @ 3DV ETH Zurich 2022****

Students: **Liyuan Zhu, Jingyan Li, Han Sun**

Supervisors: **[Iro Armeni](https://ir0.github.io/)**, **[Daniel Barath](https://people.inf.ethz.ch/dbarath/)**

### **[Poster](https://github.com/Zhu-Liyuan/SeqLoc/blob/main/doc/poster.pdf)** | **Report**

In **SeqLoc**, we propose localization with a short image sequence to leverage the redundant information in the sequence. Instead of establishing 2D-3D correspondences, we solve the pose estimation problem by point cloud registration. The proposed method also provides a simple and efficient 3D-3D correspondence generation algorithm to solve the transformation between two SfM-based point clouds. On top of the point cloud registration based localization, we add a global bundle adjustment module to refine the pose estimate with additional constraints from the sequence.


<!-- ## Proposed pipeline(adpated from [hloc](https://github.com/cvg/Hierarchical-Localization)) -->

<p align="center">
  <img src="https://github.com/Zhu-Liyuan/SeqLoc/blob/zly/doc/poster1.png" width="750"/>
  <br ><em>Proposed pipeline(adapted from hloc)</em>
</p>

#### Generation of 3D-3D correspondences
<p align="center">
  <img src="https://github.com/Zhu-Liyuan/SeqLoc/blob/zly/doc/poster2.png" width="400"/>
</p>


#### Localization by Point Cloud Registration([TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus))
<p align="center">
  <img src="https://github.com/Zhu-Liyuan/SeqLoc/blob/zly/doc/poster3.png" width="400"/>
</p>

## Installation
SeqLoc requires Python >=3.7, PyTorch >=1.1 and open3D==0.10.0.0

```bash
git clone --recursive https://github.com/Zhu-Liyuan/SeqLoc
cd SeqLoc/
python -m pip install -e .
```

Then we install TEASER++ with pybind
```bash
cd third_party/TEASER-plusplus
sudo apt install cmake libeigen3-dev libboost-all-dev
mkdir build && cd build
cmake -DTEASERPP_PYTHON_VERSION=3.6 .. && make teaserpp_python
cd python && pip install .
```
Then build [colmap](https://github.com/colmap/colmap) and modified pycolmap from source
```bash
cd ../../third_party/pycolmap
pip install .
```

## Run **_seqloc_** Pipeline

You can run the _seqloc_ pipeline **step by step** by using the following commands:

1. SFM to reconstruct local point cloud for image sequences: 
```bash
python -m pcr.sfm_pipeline --proj_dir="PATH_TO_QUERY_FOLDER"
```

2. Extract 2d-to-3d correspondence: 
```bash
python -m pcr.pairs_3d_from_2d --db_model="PATH_TO_DATABASE_FOLDER" --query_model="PATH_TO_QUERY_FOLDER"
```


3. Register local point cloud to the global model: 
```bash
python -m pcr.poses_from_pcr --db_model="PATH_TO_DATABASE_FOLDER" --query_model="PATH_TO_QUERY_FOLDER"
```

Or you can run the _seqloc_ pipeline **as a whole** by calling:
```bash
python -m pcr.seqloc_pipeline --config="pcr/config/seqloc.yaml"
```
**Don't forget to change the Path in [seqloc.yaml](https://github.com/Zhu-Liyuan/SeqLoc/blob/zly/pcr/config/seqloc.yaml) !!!**

## Data Structure for database

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

## Some useful [python scripts](https://github.com/colmap/colmap/tree/dev/scripts/python)  to manipulate colmap database and project files 
[database.py](https://github.com/colmap/colmap/blob/dev/scripts/python/database.py) - To manipulate colmap database. 

[visualize_model.py](https://github.com/colmap/colmap/blob/dev/scripts/python/visualize_model.py) - Contains visualization function and defines a class for colmap data(points3D,cameras,images) 

[read_write_model.py](https://github.com/Zhu-Liyuan/3DV/blob/master/hloc/utils/read_write_model.py) - Parses (points3D,cameras,images) into numpy data structures.

More info about the data structure of colmap can be found at https://colmap.github.io/format.html

## Contact
Feel free to contact us if you are interested in our project or have any questions.

Liyuan Zhu liyzhu@ethz.ch

Jingyan Li jingyli@ethz.ch

Han Sun hansun@ethz.ch
