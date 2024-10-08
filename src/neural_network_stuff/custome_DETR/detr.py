import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

from src.neural_network_stuff.custome_DETR.backbone import build_backbone
from src.neural_network_stuff.custome_DETR.transformer import build_transformer
from src.neural_network_stuff.custome_DETR.matcher import build_matcher

from src.neural_network_stuff.custome_DETR.misc_stuff import NestedTensor, get_world_size, is_dist_avail_and_initialized, nested_tensor_from_tensor_list, accuracy, interpolate 
from src import configure
#from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
#                           dice_loss, sigmoid_focal_loss)


transform = transforms.Compose([
        transforms.ToTensor(),  # Konvertiert das Bild in einen Tensor
        transforms.Resize((840, 960)),  # Ändert die Größe des Bildes
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisiert die Pixelwerte
    ])

class CustomeDetrModel(nn.Module):
    def __init__(self, num_classes=5, num_queries=20):
            super().__init__()
            self.backbone = build_backbone()
            self.transformer = build_transformer()
            hidden_dim = self.transformer.d_model       

            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

            self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
            self.linear_data = nn.Linear(hidden_dim, 3)


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        # Backbone and Position Stuff
        features, pos = self.backbone(samples)


        src, mask = features[-1].decompose()
        assert mask is not None


        hs_d = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        hs = hs_d['output']
        attention = hs_d['attention']

        output_center_degree_points = self.linear_data(hs)
        outputs_class = self.linear_class(hs).sigmoid()




        attn_weights = []
        for decoder_layer in self.transformer.decoder.layers:
            attn_weights.append(attention)

        #degrees = [int(360/64*deg) for deg in degrees]
        out = {'outputs_class': outputs_class[-1], 'output_center_degree_points': output_center_degree_points[-1], 'attention_weights': attn_weights}
        return out











class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        
        assert 'outputs_class' in outputs, 'no define gusto outputo what are you doing little man'
        src_logits = outputs['outputs_class']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["classes"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        #print(src_logits.shape,target_classes.shape)
        loss_ce = nn.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=0)
        losses = {'loss_ce': loss_ce}

        #print(src_logits)
        #print(target_classes)
        #print(f'loss_cd: {loss_ce}')

        #print('------------Class LOSS-----------------')
        #print(f'Output: {src_logits}')
        #print(f'Target: {target_classes_o}')
        #print(f'Data_loss: {loss_ce}')
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    

    def loss_center_degree(self, outputs, targets, indices, num_points):
        assert 'output_center_degree_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_cds = outputs['output_center_degree_points'][idx]
        target_cds = torch.cat([t['data'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Dass die 0 Objekte nicht in Loss mit reinfließen
        ignore_condition = torch.tensor([[a != 0 for a in t['classes']] for t in targets])
        ignore_condition = ignore_condition.view(-1)

        # Create a mask for positions to ignore  ignore condition == not Ignore condition
        not_ignore_ignore_mask = (ignore_condition != 0)
        # Compute L1 loss only for positions not to be ignored
        loss_cd = nn.functional.l1_loss(src_cds, target_cds, reduction='none')
        #print(f'loss_cd: {len(loss_cd)}')
        loss_cd = loss_cd[not_ignore_ignore_mask]
        #print(f'loss_cd: {len(loss_cd)}')

        losses = {}
        losses['loss_cd'] = loss_cd.sum() / num_points
        return losses


    # Benutze ich nicht
    def loss_masks(self, outputs, targets, indices, num_points):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        #losses = {
        #    "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
        #    "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        #}
        #return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'center_degree': self.loss_center_degree,
            #'masks': self.loss_masks
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        #print(f'outputs gesamt:{outputs}')
        #print(f'target gesamt{targets}')
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_points = sum(len(t["classes"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points))
        
        return losses
    



def build():

    # Args stuff
    cD_loss_coef = configure.cD_loss_coef
    ce_loss_coef = configure.ce_loss_coef
    giou_loss_coef = configure.giou_loss_coef
    eos_coef = configure.eos_coef
    num_classes = configure.num_classes
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    device = torch.device(device)

    model = CustomeDetrModel(
        num_classes=num_classes,
        num_queries=20,
    )

    matcher = build_matcher()
    weight_dict = {'loss_ce': ce_loss_coef, 'loss_cD': cD_loss_coef}
    weight_dict['loss_giou'] = giou_loss_coef


    losses = ['labels', 'center_degree']

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=eos_coef, losses=losses)
    criterion.to(device)
    return model, criterion


