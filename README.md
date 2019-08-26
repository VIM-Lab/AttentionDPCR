# AttentionDPCR
This repository contains the source codes for the paper "Attention-Based Dense Point Cloud Reconstruction from a Single Image"

# Overview
3D Reconstruction has drawn much attention in computer vision. Generating a dense point cloud from a single image is a more challenging task. However, generating dense point clouds directly costs expensively in calculation and memory and may cause the network hard to train. In this work, we propose a two-stage training dense point cloud generation network. We first train our attention-based sparse point cloud generation network to generate a sparse point cloud from a single image. Then we train our dense point cloud generation network to densify the generated sparse point cloud. After combining the two stages and finetuning, we obtain an end-to-end network that generates a dense point cloud from a single image. Through evaluation of both synthetic and real-world datasets, we demonstrate that our approach outperforms state of the art works in dense point cloud generation.
![overview](https://github.com/VIM-Lab/AttentionDPCR/blob/master/images/overview_network.jpg)

# Dataset
We train and validate our model on the ShapeNet dataset. The code of data preprocessing is released and the guideline is coming soon.

# Setup
Before get started, you may need to install numpy, tensorflow-gpu-1.4, python-opencv etc. by pip or sudo apt-get. Then you need to complie some CUDA code.
```
$ cd metric
$ make

$ cd ..\model\layer\tf_ops
$ cd grouping
$ bash tf_grouping_compile.sh
$ cd ..\interpolating
$ bash tf_interpolating_compile.sh
$ cd ..\sampling
$ bash tf_sampling_compile.sh

```

# Training
Our network is a two-stage training network.
* To train sparse point cloud generation network, run:
```
python train_sparse.py sparse
```
* To train dense point cloud generation network, run:
```
python train_dense.py dense
```
* To finetune two stages, run:
```
python train_finetune.py finetune
```

# Visualization
**ShapeNet**<br>
Below are a few sample reconstructions from our trained model tested on ShapeNet. 
![shapenet](https://github.com/VIM-Lab/AttentionDPCR/blob/master/images/qualitative_result_shapenet.jpg)

**Pix3D**<br>
Below are a few sample reconstructions from our trained model tested on real-world Pix3D dataset. Note that we mask out the background using the provided masks before passing the images through the network. 
![pix3d](https://github.com/VIM-Lab/AttentionDPCR/blob/master/images/qualitative_result_pix3d.jpg)
