import math
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from scipy.optimize import linear_sum_assignment
from torchvision.models._utils import IntermediateLayerGetter


from DETR_from_scratch import FrozenBatchNorm2d
from positional import PositionEmbeddingSine

class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                train_backbone = False,
                pretrained = True,
                return_interm_layers = False,
                dilation = False,
                norm_layer = None,
                hidden_dim = 256):
        super().__init__()
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation = [False, False, dilation],
            pretrained = pretrained,
            norm_layer = norm_layer or FrozenBatchNorm2d
        )
        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        N_steps = hidden_dim // 2
        self.position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    def forward(self, x, mask = None):
        mask = mask or torch.zeros_like(x[:,0], dtype = torch.bool)
        xs = self.body(x)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            pos = self.position_embedding(x, m).to(x.dtype)
            out[name] = (x, m, pos)
        return out[self.body.return_layers["layer4"]]