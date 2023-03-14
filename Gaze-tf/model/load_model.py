"""
为了方便切换加载五个模型，在此对模型加载进行封装


"""

from model import vit
from model import swin
from model import aff_model
from model import gazetr_model
from model import new_model


def loadTheModel(config):
    model = []
    model_name = config['model_name']
    if model_name == "aff":
        model = aff_model.model()

    if model_name == "vit":
        image_size = config['image_size']
        patch_size = config['patch_size']
        num_classes = config['num_classes']
        dim = config['dim']
        depth = config['depth']
        heads = config['heads']
        mlp_dim = config['mlp_dim']
        dropout = config['dropout']
        emb_dropout = config['emb_dropout']
        model = vit.ViT(image_size=image_size,
                        patch_size=patch_size,
                        num_classes=num_classes,
                        dim=dim,
                        depth=depth,
                        heads=heads,
                        mlp_dim=mlp_dim,
                        dropout=dropout,
                        emb_dropout=emb_dropout)

    if model_name == "gazetr":
        model = gazetr_model.Model()

    if model_name == "new_model":
        model = new_model.Model()

    if model_name == "swin":
        image_size = config['image_size']
        patch_size = config['patch_size']
        num_classes = config['num_classes']
        embed_dim = config['embed_dim']
        depths = config['depths']
        num_heads = config['num_heads']
        window_size = config['window_size']
        drop_rate = config['drop_rate']
        drop_path_rate = config['drop_path_rate']
        in_chans = config['in_chans']
        model = swin.SwinTransformer(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            # mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            # qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            # qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ape=False,
            patch_norm=True,
            use_checkpoint=False)

    return model
