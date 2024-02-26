# Learning Virtual View Selection for 3D Semantic Segmentation
A pytorch implement of The paper “Learning Virtual View Selection for 3D Scene Semantic Segmentation”

## introduction

we propose a general learning framework for joint 2D-3D scene understanding by selecting informative virtual 2D views of the underlying 3D scene. We feed both the 3D geometry and the generated virtual 2D views into any joint 2D-3D-input or pure 3D-input based deep neural models for improving 3D scene understanding. 

We have validated our proposed framework for various joint 2D-3D-input or pure 3D-input based deep neural models on ScanNet v2 and S3DIS, and the results demonstrate that our method obtains a consistent gain over baseline models and achieves new top accuracy for scene semantic segmentation.

The repo contains 4 subprojects to show the application of our work on different existing segmentation methods.

## Environment


```bash
# Torch
$ pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
# MinkowskiEngine 0.4.1
$ conda install numpy openblas
$ git clone https://github.com/StanfordVL/MinkowskiEngine.git
$ cd MinkowskiEngine
$ git checkout f1a419cc5792562a06df9e1da686b7ce8f3bb5ad
$ python setup.py install
# Others
$ pip install imageio==2.8.0 opencv-python==4.2.0.32 pillow==7.0.0 pyyaml==5.3 scipy==1.4.1 sharedarray==3.2.0 tensorboardx==2.0 tqdm==4.42.1
```

for each subproject, refer to their readme for details

## Prepare data

- Download the dataset Scannet and S3DIS on their official website. http://www.scan-net.org/ & http://buildingparser.stanford.edu/dataset.html

