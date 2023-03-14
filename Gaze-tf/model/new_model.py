"""
new model about 2d gaze
input: leye reye face origin
output: position on screen


"""
import torch
import torch.nn as nn
import numpy as np
import math
import copy
from model import resnet
# import resnet


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos

    def forward(self, src, pos):
        # src_mask: Optional[Tensor] = None,
        # src_key_padding_mask: Optional[Tensor] = None):
        # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class FaceImageEncoder(nn.Module):

    def __init__(self) -> None:
        super(FaceImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=0),
            nn.GroupNorm(6, 48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=0),
            nn.GroupNorm(12, 96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EyeImageEncoder(nn.Module):

    def __init__(self) -> None:
        super(EyeImageEncoder, self).__init__()
        maps = 32
        nhead = 8
        dim_feature = 7 * 7
        dim_feedforward = 512
        dropout = 0.1
        num_layers = 6
        self.imageHandle = resnet.resnet18(pretrained=False, maps=maps)
        eye_encoder_layer = TransformerEncoderLayer(maps, nhead,
                                                    dim_feedforward, dropout)
        eye_encoder_norm = nn.LayerNorm(maps)
        self.eye_encoder = TransformerEncoder(eye_encoder_layer, num_layers,
                                              eye_encoder_norm)
        self.eye_cls_token = nn.Parameter(torch.randn(1, 1, maps))
        self.pos_embedding = nn.Embedding(dim_feature + 1, maps)

    def forward(self, eyeImage):
        # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        eye_feature = self.imageHandle(eyeImage)
        batch_size = eye_feature.size(0)
        eye_feature = eye_feature.flatten(2)
        eye_feature = eye_feature.permute(2, 0, 1)
        eye_cls = self.eye_cls_token.repeat((1, batch_size, 1))
        eye_feature = torch.cat([eye_cls, eye_feature], 0)
        position = torch.from_numpy(np.arange(0, 50)).cuda()
        pos_feature = self.pos_embedding(position)
        eye_feature = self.eye_encoder(eye_feature, pos_feature)
        eye_feature = eye_feature.permute(1, 2, 0)
        eye_feature = eye_feature[:, :, 0]
        return eye_feature


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.leye = EyeImageEncoder()
        self.reye = EyeImageEncoder()
        self.face = FaceImageEncoder()
        self.origin = FaceImageEncoder()
        self.fc = nn.Sequential(
            nn.Linear(4 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )

    def forward(self, batch, device):
        # x_face torch.Size([10, 3, 224, 224])
        x_face = batch["faceImg"]
        x_leye = batch["leftEyeImg"]
        x_reye = batch["rightEyeImg"]
        x_origin = batch["origin"]
        leye_feature = self.leye(x_leye)
        reye_feature = self.reye(x_reye)
        face_feature = self.face(x_face)
        origin_feature = self.origin(x_origin)
        feature = torch.cat(
            (leye_feature, reye_feature, face_feature, origin_feature), 1)
        # print(feature.shape)
        gaze = self.fc(feature)
        return gaze

    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        loss = self.loss_op(gaze, label)
        return loss


if __name__ == "__main__":
    # config_path = "configs/test_config.yaml"
    # config  = yaml.load(config_path)
    m = Model()
    feature = {
        "faceImg": torch.zeros(10, 3, 224, 224),
        "leftEyeImg": torch.zeros(10, 3, 224, 224),
        "rightEyeImg": torch.zeros(10, 3, 224, 224),
        "origin": torch.zeros(10, 3, 224, 224),
        "label": torch.zeros(10, 2),
        "frame": "test.jpg"
    }
    a = m(feature)
    print(a.shape)
