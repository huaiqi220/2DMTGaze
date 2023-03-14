# ReadMe

our work multi-input-transformer MITF


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
new_model: our work



## benchmark
all result on VGSP 5-Fold

