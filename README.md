# 3DV @ Group 16
Image based localization and map update in a quality adaptive manner. 

Group members: **Jingyan Li, Liyuan Zhu, Han Sun**
## Prospective pipeline
<img src="https://github.com/Zhu-Liyuan/3DV/blob/master/doc/3dv.png" height="600"/>


### Some useful [python scripts](https://github.com/colmap/colmap/tree/dev/scripts/python)  to manage colmap database and project files 
[database.py](https://github.com/colmap/colmap/blob/dev/scripts/python/database.py) - To manipulate colmap database. 

[visualize_model.py](https://github.com/colmap/colmap/blob/dev/scripts/python/visualize_model.py) - Contains visualization function and defines a class for colmap data(points3D,cameras,images) 

[read_write_model.py](https://github.com/Zhu-Liyuan/3DV/blob/master/hloc/utils/read_write_model.py) - Parses (points3D,cameras,images) into numpy data structures.

More info about the data structure of colmap can be found at https://colmap.github.io/format.html
