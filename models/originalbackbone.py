from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding
from dilation_transformer import Dilation_Transformer

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, backbone_name: str, return_interm_layers: bool, pretrained=""):
        super().__init__()
        if "dilation" in backbone_name:
            for name, parameter in backbone.named_parameters():
                if 'absolute_pos_embed' in name or 'relative_position_bias_table' in name or 'norm' in name:
                    parameter.requires_grad_(False)
            print("load pretrained model...")
            if pretrained != "":
                backbone.init_weights(pretrained)
                # param = torch.load(pretrained)['model']
                # backbone.load_state_dict(param)
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    def __init__(self, backbone_name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 pretrained=None,
                 use_checkpoint=False,
                 dilation=False):
        if "dilation" in backbone_name:
            if 'small' in backbone_name:
                backbone = Dilation_Transformer(depths=[2, 2, 18, 2], dilation_rate_list=[1, 2, 3])
            elif 'base' in backbone_name:
                if "384" in backbone_name:
                    backbone = Dilation_Transformer(pretrain_img_size=384, depths=[2, 2, 18, 2],
                                                    dilation_rate_list=[1, 2, 3], embed_dim=128,
                                                    num_heads=[4, 8, 16, 32],
                                                    use_checkpoint=use_checkpoint)
                else:
                    backbone = Dilation_Transformer(depths=[2, 2, 18, 2], dilation_rate_list=[1, 2, 3], embed_dim=128,
                                                    num_heads=[4, 8, 16, 32],
                                                    use_checkpoint=use_checkpoint)
            elif 'large' in backbone_name:
                if "384" in backbone_name:
                    backbone = Dilation_Transformer(pretrain_img_size=384, depths=[2, 2, 18, 2],
                                                    dilation_rate_list=[1, 2, 3], embed_dim=192,
                                                    num_heads=[6, 12, 24, 48], window_size=12,
                                                    use_checkpoint=use_checkpoint)
                else:
                    backbone = Dilation_Transformer(depths=[2, 2, 18, 2], dilation_rate_list=[1, 2, 3], embed_dim=192,
                                                    num_heads=[6, 12, 24, 48],
                                                    use_checkpoint=use_checkpoint)
            else:
                backbone = Dilation_Transformer(depths=[2, 2, 6, 2], dilation_rate_list=[1, 2, 3],
                                                use_checkpoint=use_checkpoint)
        else:
            norm_layer = FrozenBatchNorm2d
            backbone = getattr(torchvision.models, backbone_name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=norm_layer)
            assert backbone_name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, backbone_name, pretrained)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
