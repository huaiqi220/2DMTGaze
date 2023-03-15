# 2DMTGaze: 2D Point of Gaze Multi-Stream Model Using Transformer

our work 2DMTGaze for 2D Point of Gaze Estimation 

## Abstract
Transformers have achieved better performance in most computer vision tasks than models based on convolutional neural networks.
The 2D Point of Gaze estimation task can predict the coordinates of the human eye's focus point on the plane through the user's facial image, which is a subproblem in the field of gaze estimation and can be applied to smartphones, laptops, televisions, and other smart devices.
Previous work has verified the performance of the transformer in the 3D gaze estimation, but the performance of transformers in the 2D PoG Estimation task remains to be studied.
We propose 2D multi-stream gaze estimation model 2DMTGaze (2D Multi-stream Transformer Gaze) 
based on Transformer.
We compare the performance of 2DMTGaze with the popular Vision-transformer, Swin-transformer, and the advanced convolution-based gaze method AFF-Net.
The experiment shows that the original unoptimized transformer performs poorly on the 2D PoG task. 
The 2DMTGaze combines facial and eye features and performs the same as the convolution-based model on the benchmark.

<img width="1826" alt="截屏2023-03-15 12 19 17" src="https://user-images.githubusercontent.com/64659513/225205547-8179ab23-cec2-4f78-80f7-7479793a0093.png">


## Compared model
Vit: vision transformer
    image_size: 224
    patch_size: 32
    num_classes: 2
    dim: 1024
    depth: 6
    heads: 16
    mlp_dim: 2048
    dropout: 0.1
    emb_dropout: 0.1

Swin: Swin transformer
    image_size: 224
    patch_size: 4
    num_classes: 2
    embed_dim: 128
    depths: [ 2, 2, 18, 2 ]
    num_heads: [ 4, 8, 16, 32 ]
    window_size: 7
    drop_rate: 0
    drop_path_rate: 0.1
    in_chans: 3

GazeTR: GazeTR
    GazeTR-Hybrid

aff: aff-net

iTracker: iTracker

new_model: our work



## benchmark
all result on MPIIFaceGaze and GazeCapture
<img width="797" alt="截屏2023-03-15 12 17 30" src="https://user-images.githubusercontent.com/64659513/225205296-1da1a7fa-07f9-4d7c-b1c1-0bd2d26f2246.png">

<img width="1072" alt="截屏2023-03-15 12 19 00" src="https://user-images.githubusercontent.com/64659513/225205559-56e13a23-e293-4207-bba9-ed4aa5607d20.png">

