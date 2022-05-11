# The CMU Seasons Dataset
This is the public release of the CMU Seasons dataset that is used in this paper to 
benchmark visual localization and place recognition algorithms under changing conditions:
```
T. Sattler, W. Maddern, C. Toft, A. Torii, L. Hammarstrand, E. Stenborg, D. Safari, M. Okutomi, M. Pollefeys, J. Sivic, F. Kahl, T. Pajdla. 
Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions. 
Conference on Computer Vision and Pattern Recognition (CVPR) 2018 
```
The dataset is based on the CMU Visual Localization dataset described here:
```
Hernan Badino, Daniel Huber, and Takeo Kanade. 
The CMU Visual Localization Data Set. 
http://3dvis.ri.cmu.edu/data-sets/localization, 2011.
```
The CMU Seasons dataset uses a subset of the images provided in the CMU Visual 
Localization dataset. It uses images taken under a single reference condition (`sunny + no foliage`), 
captured at 17 locations (referred to as *slices* hereafter), to represent the scene. For this 
reference condition, the dataset provides a reference 3D model reconstructed using 
Structure-from-Motion. The 3D model consequently defines a set of 6DOF reference poses 
for the database images. In addition, query images taken under different conditions at the 17 
slices are provided. For these query images, the reference 6DOF poses will not be released.



## License
The CMU Seasons dataset builds on top of the CMU Visual Localization datset created by 
Hernan Badino, Daniel Huber, and Takeo Kanade. The original dataset can be found 
[here](http://3dvis.ri.cmu.edu/data-sets/localization/). 
The images provided with the CMU Seasons dataset originate from the CMU Visual 
Localization dataset. They are licensed under a 
[Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/) license (see also [here](http://3dvis.ri.cmu.edu/data-sets/localization/) under "License"). 
As the files provides by the CMU Seasons dataset build upon the original material from the 
CMU Visual Localization dataset, they are also licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/) license.

Please see this [webpage](http://3dvis.ri.cmu.edu/data-sets/localization/) if you are
interested in using the images commercially.

### Using the CMU Seasons Dataset
By using the CMU Seasons dataset, you agree to the license terms set out above.
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
The CMU Seasons dataset uses images captured by two camers mounted on a car. The 
images provided with the CMU Seasons dataset have been undistorted.
Below is a table mapping the slices to different scenarios. In addition, we provide a list of the 
different conditions and their corresponding capture dates listed on the CMU Visual Localization 
[webpage](http://3dvis.ri.cmu.edu/data-sets/localization/). An extended 
version of these tables is included in the extended version of the CVPR 2018 paper that can be
found on [arXiv](https://arxiv.org/abs/1707.09092).

Scene | Slices
------------|----------------
Urban | 2-8
Suburban | 9-10, 17
Park | 18-22, 24-25

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
* `3D-models/`: Contains the 3D models created from the reference images.
* `intrinsics/`: Contains the intrinsic calibrations of the two cameras used in the dataset.
* `query_lists/`: Contains text files with the list of query images per slice.

In addition, the file `images.zip` contains the database and query images for each slice. 
The SIFT descriptors [1] for these images used for the evaluation in the CVPR 2018 paper are 
provided in the `sift_descriptors.zip` file.
 
In the following, we will describe the different files provided with the dataset in more detail.

### 3D Models
This directory contains the reference 3D models build for the dataset. We provide a 3D model 
per slice. For each slice, we provide its relevant query images.

We provide the 3D reconstructions in various formats:
* `NVM models` (`.nvm` file names): 3D models in the [NVM file format](http://ccwu.me/vsfm/doc.html#nvm) 
used by VisualSfM [2,3]. 
* `Bundler models` (`.out` and `.list.txt` file names): 3D models in the [file format](http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6) 
used by Bundler [4]. The `.list.txt` file for a given model, e.g., `slice3.list.txt` 
for the `slice3.out` model, specifies the images in the reconstruction. The order is the 
same as in the reconstruction. 
* Binary `.info` files: These are binary files containing the poses of all database images, all 
3D points in the reference 3D model, and their corresponding descriptors. A C++ code 
snippet for loading the data is available [here](https://s3.amazonaws.com/LocationRecognition/Datasets/load_info_file_snippet.cc).

Please familiarize yourself with the different file formats. Please also pay special attention to 
the different camera coordinate systems and conventions of the different formats described below. 

#### Coordinate Systems and Conventions (Important!)
We provide the 3D models in different file formats for convenience. However, providing 
different formats also comes with a catch: *Different formats use different coordinate systems 
and conventions*.

##### Camera Coordinate Systems
The NVM models use the camera coordinate system typically used in *computer vision*. 
In this camera coordinate system, the camera is looking down the `z`-axis, with the `x`-axis 
pointing to the right and the `y`-axis pointing downwards. The coordinate `(0, 0)` 
corresponds to the top-left corner of an image. 

The Bundler models and the `.info` files (which are based on the Bundler models) use the 
same camera coordinate system, but one that differs from the *computer vision camera 
coordinate system*. More specifically, they use the camera coordinate system typically used 
in *computer graphics*. In this camera coordinate system, the camera is looking down the 
`-z`-axis, with the `x`-axis pointing to the right and the `y`-axis pointing upwards. The 
coordinate `(0, 0)` corresponds to the lower-left corner of an image. 
Camera poses in the Bundler and `.info` formats can be converted to poses in the NVM 
formats via the following pseudo-code
```
// Here, R_* is the rotation from the world into the camera coordinate system
// and c_* is the position of the camera in the world-coordinate system.
// We denote the two coordinate systems as "nvm" (NVM and COLMAP) and 
// "out" (Bundler and .info files).
c_nvm = c_out;
// Mirrors the y- and z-coordinates
c_nvm[1] *= -1.0;
c_nvm[2] *= -1.0;
R_nvm = R_out;
// Changes the sign of some entries.
R_nvm(0, 1) *= -1.0;
R_nvm(0, 2) *= -1.0;
R_nvm(1, 0) *= -1.0;
R_nvm(2, 0) *= -1.0;
```
For the **evaluation** of poses estimated on the CMU Seasons dataset, you will need to 
provide **pose estimates in the coordinate system used by NVM**.

##### Conventions
The different types of models store poses in different formats.
* The NVM models store the rotation (as a quaternion) from the world coordinate system 
to the camera coordinate system as well as the camera position in world coordinates. Thus, 
NVM stores a pose as `R, c` and the translation of the camera can be computed as 
`t = -(R * c)`.
* The Bundler models and `.info` files store the rotation (as a matrix) from the world 
coordinate system to the camera coordinate system as well as the camera translation. Thus, 
they store a pose as `[R|t]`.

We strongly recommend that you familiarize yourself with the file format of the models that 
you plan to use.

### Intrinsics
The intrinsic calibration of the two cameras used to capture the dataset's images. 
For each camera, we store the intrinsic calibration matrix as 
```
fx  0 cx
 0 fy cy
 0  0  1
```
Here, `fx`  and `fy` are the focal lengths for the two axes and `cx` and `cy` specify the position 
of the principal point. Since all images used in the dataset have been undistorted, the 
cameras can be modeled by the pinhole camera model. The coordinate `(0, 0)` corresponds 
to the top-left corner of an image, with the x-axis pointing to the right and the y-axis pointing 
downwards. The type of camera used to capture an image is evident from the file name of the 
image (containing either `_c0_` for camera 0 or `_c1`_ for camera 1).

### Query Lists
For each slice, we provide a text file with the list of query images for that slice. The text file 
stores a line per query image. Here is an example from slice 2:
```
query/img_01555_c1_1283347879534213us_rect.jpg PINHOLE 1024 768 873.38 876.49 529.32 397.27
```
Here, `img_01555` indicates that this image is the 1555th image in the dataset. The infix 
`_c1_` indicates that camera 1 was used to capture the image. `1283347879534213us` is the 
timestamp of the capture time. `_rect` indicates that the image has been rectified. 
`PINHOLE` indicates that the image can be modelled by a pinhole camera. `1024 768` are the 
dimensionality (width and height) of the image. `873.38 876.49 529.32 397.27` specify 
the focal lengths and principal point in the format `fx fy cx cy`.

### SIFT Descriptors
The SIFT descriptors are provided in the  `sift_descriptors.zip` file. They are stored in 
the binary file format used by [VisualSfM](http://ccwu.me/vsfm/index.html) as described 
[here](http://ccwu.me/vsfm/doc.html#customize). Here is a C++ code snippet for loading the 
descriptors:
```
struct Keypoint {
  float x;
  float y;
  float scale;
  float orientation;
};

bool LoadVisualSFMFeatures(const char *filename,
                           std::vector<Keypoint>* keypoints,
                           std::vector<unsigned char*>* descriptors) {
  // Assumes that the output vectors are empty.
  // See http://www.cs.washington.edu/homes/ccwu/vsfm/doc.html#customize
  // for a description of the file format.

  // Initializes constants.   
  int siftname = ('S' + ('I'<<8) + ('F'<<16) + ('T'<<24));
  int sift_version4 = ('V'+('4'<<8)+('.'<<16)+('0'<<24));
  int sift_version5 = ('V'+('5'<<8)+('.'<<16)+('0'<<24));
  int sift_eof_marker = (0xff+('E'<<8)+('O'<<16)+('F'<<24));

  std::ifstream ifs(filename, std::ios::in | std::ios::binary);

  if (!ifs.is_open()) {
    std::cerr << "ERROR: Could not open file " << filename << std::endl;
    return false;
  }

  // Reads the header.
  int name, version, num_keypoints, num_keypoint_entries, desc_size;

  ifs.read((char*) &name, sizeof(int));
  if(name != siftname) {
    std::cerr << "ERROR: The name " << name
              << " is not the expected name " << siftname << std::endl;
    ifs.close();
    return false;
  }

  ifs.read((char*) &version, sizeof(int));
  if(!(version == sift_version4 || version == sift_version5)) {
    std::cerr << "ERROR: Unsupported version " << version << std::endl;
    ifs.close();
    return false;
  }

  ifs.read((char*) &num_keypoints, sizeof(int));
  ifs.read((char*) &num_keypoint_entries, sizeof(int));
  ifs.read((char*) &desc_size, sizeof(int));
  
  if (num_keypoint_entries < 4 || num_keypoint_entries > 5) {
    std::cerr << "ERROR: Expected 4 or 5 entries, not " 
              << num_keypoint_entries << std::endl;
  }
  
  if(desc_size != 128) {
    std::cerr << "ERROR: Number of descriptor entries " 
              << desc_size << " != 128" << std::endl;
    ifs.close();
    return false;
  }

  // Loads the keypoints
  keypoints->resize(num_keypoints);
  descriptors->resize(num_keypoints);
  
  float color;
  for(int i=0; i < num_keypoints; ++i) {
    ifs.read((char*) &((*keypoints)[i].x), sizeof(float));
    ifs.read((char*) &((*keypoints)[i].y), sizeof(float));
    if(num_keypoint_entries == 5) {
      // Loads the color, but ignores it.
      ifs.read((char*) &color, sizeof(float));
    }
    ifs.read((char*) &((*keypoints)[i].scale), sizeof(float));
    ifs.read((char*) &((*keypoints)[i].orientation), sizeof(float));
  }

  // Load the descriptors.
  for(int i=0; i < num_keypoints; ++i) {
    (*descriptors)[i] = new unsigned char[128];
    ifs.read((char*) (*descriptors)[i], sizeof(unsigned char) * 128);
  }

  // Check if all data was loaded properly, i.e., if we have reached the
  // end of the file.
  int eof_marker;
  ifs.read((char*) &eof_marker, sizeof(int));

  ifs.close();

  if(eof_marker != sift_eof_marker) {
    std::cout << "ERROR: Wrong eof_marker " << eof_marker << std::endl;
    return false;
  }
  return true;
}

```

## Evaluating Results on the CMU Seasons Datset
We are currently setting up an [evaluation service](http://visuallocalization.net/) for the 
benchmarks proposed in the CVPR 2018 paper. You will be able to upload the poses 
estimated by your method to this service, which in turn will provide results to you. While the 
service is being set-up, we will manually evaluate the results for you. To use this service, 
please send your results via email to [Torsten Sattler](mailto:sattlert@inf.ethz.ch).

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

Please adhere to the following naming convention for the files that you submit:
```
CMU_eval_[yourmethodname].txt
```
Here, `yourmethodname` is some name or identifier chosen by yourself. This name or identifier 
should be as unique as possible to avoid confusion with other methods. Once the evaluation 
service is ready, it will be used to display the results of your method.

**IMPORTANT:** Our evaluation tools expect that the coordinate system in which the camera 
pose is expressed is the **NVM coordinate system**. If you are using the Bundler or `.info` 
coordinate system to estimate poses, you will need to **convert poses to the NVM 
coordinate system** before submission. 
A good **sanity check** to ensure that you are submitting poses in the correct format is to 
query with a reference image and then check whether the pose matches the reference pose 
defined in the NVM model after converting the stored camera  position to a translation (as 
described above).


## References:
1. D. Lowe. Distinctive Image Features from Scale-Invariant Keypoints. International Journal 
of Computer Vision (IJCV) 2004.
2. C. Wu. Towards Linear-time Incremental Structure From Motion. 3DV 2013
3. C. Wu. VisualSFM: A Visual Structure from Motion System. http://ccwu.me/vsfm/, 2011
4. N. Snavely, S. M. Seitz, R. Szeliski. Photo Tourism: Exploring image collections in 3D. 
ACM Transactions on Graphics (Proceedings of SIGGRAPH 2006) 2006.
