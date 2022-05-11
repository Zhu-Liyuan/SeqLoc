# The Extended CMU Seasons Dataset
This is the public release of the Extended CMU Seasons dataset that is used in this paper to 
benchmark visual localization and place recognition algorithms under changing conditions:
```
This is an extended version of the dataset published in 
T. Sattler, W. Maddern, C. Toft, A. Torii, L. Hammarstrand, E. Stenborg, D. Safari, M. Okutomi, M. Pollefeys, J. Sivic, F. Kahl, T. Pajdla. 
Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions. 
Conference on Computer Vision and Pattern Recognition (CVPR) 2018 
```
This dataset is in turn based on the CMU Visual Localization dataset described here:
```
Hernan Badino, Daniel Huber, and Takeo Kanade. 
The CMU Visual Localization Data Set. 
http://3dvis.ri.cmu.edu/data-sets/localization, 2011.
```
The Extended CMU Seasons dataset uses a subset of the images provided in the CMU Visual 
Localization dataset. It uses images taken under a single reference condition (`sunny + no foliage`), 
captured at 17 locations (referred to as *slices* hereafter), to represent the scene. For this 
reference condition, the dataset provides a reference 3D model reconstructed using 
Structure-from-Motion. The 3D model consequently defines a set of 6DOF reference poses 
for the database images. In addition, query images taken under different conditions at the 17 
slices are provided. Reference poses for around 50% of these images are included in this dataset, 
in addition to the reference sequence as described above. 

## License
The Extended CMU Seasons dataset builds on top of the CMU Visual Localization dataset created by 
Hernan Badino, Daniel Huber, and Takeo Kanade. The original dataset can be found 
[here](http://3dvis.ri.cmu.edu/data-sets/localization/). 
The images provided with the CMU Seasons dataset originate from the CMU Visual 
Localization dataset. They are licensed under a 
[Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/) license (see also [here](http://3dvis.ri.cmu.edu/data-sets/localization/) under "License"). 
As the files provided by the Extended CMU Seasons dataset build upon the original material from the 
CMU Visual Localization dataset, they are also licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/) license.

Please see this [webpage](http://3dvis.ri.cmu.edu/data-sets/localization/) if you are
interested in using the images commercially.

### Using the Extended CMU Seasons Dataset
By using the Extended CMU Seasons dataset, you agree to the license terms set out above.
If you are using the CMU Seasons dataset in a publication, please cite **both** of the
following two sources:
```
@inproceedings{Sattler2018CVPR,
  author={Sattler, Torsten and Maddern, Will and Toft, Carl and Torii, Akihiko and Hammarstrand, Lars and Stenborg, Erik and Safari, Daniel and Okutomi, Masatoshi and Pollefeys, Marc and Sivic, Josef and Kahl, Fredrik and Pajdla, Tomas},
  title={{Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions}},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018},
}

@misc{Badino2011,
  author = {Badino, Hernan and Huber, Daniel and Kanade, Takeo},
  title = {{The CMU Visual Localization Data Set}},
  year = {2011},
  howpublished = {\url{http://3dvis.ri.cmu.edu/data-sets/localization}}
}
```

### Image Details
The Extended CMU Seasons dataset uses images captured by two camers mounted on a car. 
Below is a table mapping the slices to different scenarios. In addition, we provide a list of the 
different conditions and their corresponding capture dates listed on the CMU Visual Localization 
[webpage](http://3dvis.ri.cmu.edu/data-sets/localization/). An extended 
version of these tables is included in the extended version of the CVPR 2018 paper that can be
found on [arXiv](https://arxiv.org/abs/1707.09092).

Scene | Slices
------------|----------------
Urban | 2-8
Suburban | 9-17
Park | 18-25

Condition | Capture Date
------------|----------------
Sunny + No Foliage (reference) | 4 Apr 2011
Sunny + Foliage | 1 Sep 2010
Sunny + Foliage | 15 Sep 2010
Cloudy + Foliage | 1 Oct 2010
Sunny + Foliage | 19 Oct 2010
Overcast + Mixed Foliage | 28 Oct 2010
Low Sun + Mixed Foliage | 3 Nov 2010
Low Sun + Mixed Foliage | 12 Nov 2010
Cloudy + Mixed Foliage | 22 Nov 2010
Low Sun + No Foliage + Snow | 21 Dec 2010
Low Sun + Foliage | 4 Mar 2011
Overcast + Foliage | 28 Jul 2011


### Privacy
We take privacy seriously. If you have any concerns regarding the images and other data
provided with this dataset, please [contact us](mailto:sattlert@inf.ethz.ch).



## Provided Files
The following files are provides with this release of the CMU Seasons dataset in various 
directories:
* `intrinsics.txt`: Contains the intrinsic calibrations of the two cameras used in the dataset.
* `slice[X].tar`: Contains all images, 2D features, reference 3D models and camera poses for a given submap
 
In the following, we will describe the different files provided with the dataset in more detail.

### 3D Models
This directory contains the reference 3D models build for the dataset. We provide a 3D model 
per slice. For each slice, we provide its relevant query images.

We provide the 3D reconstructions in the 'slice[X]/sparse' folders in the same format as used 
by Colmap. Please refer to 'https://colmap.github.io/format.html' for a description of this format. 

Please familiarize yourself with the different file formats. Please also pay special attention to 
the different camera coordinate systems and conventions of the different formats described below. 

##### Camera Coordinate Systems
The models use the camera coordinate system typically used in *computer vision*. 
In this camera coordinate system, the camera is looking down the `z`-axis, with the `x`-axis 
pointing to the right and the `y`-axis pointing downwards. The coordinate `(0, 0)` 
corresponds to the top-left corner of an image. 

For the **evaluation** of poses estimated on the CMU Seasons dataset, you will need to 
provide pose estimates in this coordinate system. 

### Intrinsics
Please refer to the intrinsics.txt file for a description of how the camera intrinsics are specified.
Note that non-linear distortion are present in the images. The intrinsics.txt file contains information
about this distortion. 

### Database Lists
For each slice, we provide a text file with the list of database images for that slice. The text file 
stores a line per database image. Here is an example from slice 2:
```
img_00122_c0_1303398475046031us.jpg 0.636779 0.569692 0.379586 -0.354792 -85.392177 55.210379 -2.980700
```
Here, `img_00122` indicates that this image is the 122 image in the dataset. The infix 
`_c1_` indicates that camera 1 was used to capture the image. `1283347879534213us` is the 
timestamp of the capture time. The seven numbers after the image name is the camera pose: 
the first four are the components of a rotation quaternion, corresponding to a rotation R, 
and the last three are the camera center C. The rotation R corresponds to the first 3x3 subblock
of the corresponding camera matrix, and the camera center C is related to the fourth column t of the camera matrix according to C = -R^T * t, where R^T denotes the transpose of R. 

### Query images 
A list of all query images for each slice is provided in the `test-images-sliceX.txt` files. For the 
evaluation, a .txt file should be submitted containing a list of all query images (for all slices)
in the same format as the database list, as specified above. 

Please submit your results as a text file using the following file format. For each query 
image for which your method has estimated a pose, use a single line. This line should store the
result as `name.jpg qw qx qy qz tx ty tz`. 
Here, `name` corresponds to the file name of the image. `qw qx qy qz` represents the 
**rotation** from world to camera coordinates as a **unit quaternion**. `tx ty tz` is the 
camera **translation** (**not the camera position**). 
An example, obtained using the *DenseVLAD* baseline from the CVPR 2018 paper, for such a 
line is 
```
img_09070_c0_1283348561545968us_rect.jpg -0.158009106041571 -0.081811017681865 -0.696368150506469 0.695278150270886 1327.578981972849306 175.561682163729188 -789.162390047261056
```
Note that the slice number **is not specified**. 

**IMPORTANT:** Our evaluation tools expect that the coordinate system in which the camera 
pose is expressed is the **NVM coordinate system**. If you are using the Bundler or `.info` 
coordinate system to estimate poses, you will need to **convert poses to the NVM 
coordinate system** before submission. 
A good **sanity check** to ensure that you are submitting poses in the correct format is to 
query with a reference image and then check whether the pose matches the reference pose 
defined in the NVM model after converting the stored camera  position to a translation (as 
described above).

### SIFT Descriptors
The SIFT descriptors extracted from all images and used to build the reference model are 
included in the database[X].db files. The descriptors can be extracted from this file using
the script export_sift_features.py in the top directory. The features are extracted like so:
python export_sift_features.py --database_path [PATH_TO_DB_FILE] --output_path [PATH_TO_DESIRED_FOLDER]


## References:
1. D. Lowe. Distinctive Image Features from Scale-Invariant Keypoints. International Journal 
of Computer Vision (IJCV) 2004.
2. C. Wu. Towards Linear-time Incremental Structure From Motion. 3DV 2013
3. C. Wu. VisualSFM: A Visual Structure from Motion System. http://ccwu.me/vsfm/, 2011
4. N. Snavely, S. M. Seitz, R. Szeliski. Photo Tourism: Exploring image collections in 3D. 
ACM Transactions on Graphics (Proceedings of SIGGRAPH 2006) 2006.
