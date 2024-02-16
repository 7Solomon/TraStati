import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

from neural_network_stuff.custome_DETR.backbone import build_backbone
from neural_network_stuff.custome_DETR.transformer import build_transformer
from neural_network_stuff.custome_DETR.matcher import build_matcher

from neural_network_stuff.custome_DETR.misc_stuff import NestedTensor, get_world_size, is_dist_avail_and_initialized, nested_tensor_from_tensor_list, accuracy, interpolate 

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

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        output_center_degree_points = self.linear_data(hs)
        outputs_class = self.linear_class(hs).sigmoid()

        #degrees = [int(360/64*deg) for deg in degrees]
        out = {'outputs_class': outputs_class[-1], 'output_center_degree_points': output_center_degree_points[-1]}
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

        loss_ce = nn.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_points):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_outputs = outputs['outputs_class']
        device = pred_outputs.device
        tgt_lengths = torch.as_tensor([len(v["classes"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_outputs.argmax(-1) != pred_outputs.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_center_degree(self, outputs, targets, indices, num_points):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        assert 'output_center_degree_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_cds = outputs['output_center_degree_points'][idx]
        target_cds = torch.cat([t['data'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_cd = nn.functional.l1_loss(src_cds, target_cds, reduction='none')

        losses = {}
        losses['loss_cd'] = loss_cd.sum() / num_points
        return losses

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
            'cardinality': self.loss_cardinality,
            'center_degree': self.loss_center_degree,
            'masks': self.loss_masks
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
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        #if 'aux_outputs' in outputs:
        #    for i, aux_outputs in enumerate(outputs['aux_outputs']):
        #        indices = self.matcher(aux_outputs, targets)
        #        for loss in self.losses:
        #            if loss == 'masks':
        #                # Intermediate masks losses are too costly to compute, we ignore them.
        #                continue
        #            kwargs = {}
        #            if loss == 'labels':
        #                # Logging is enabled only for the last layer
        #                kwargs = {'log': False}
        #            l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_points, **kwargs)
        #            l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
        #            losses.update(l_dict)

        return losses
    



def build():

    # Args stuff
    cD_loss_coef = 5
    giou_loss_coef = 2
    eos_coef = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_classes = 5
    device = torch.device(device)

    model = CustomeDetrModel(
        num_classes=num_classes,
        num_queries=20,
    )

    matcher = build_matcher()
    weight_dict = {'loss_ce': 1, 'loss_cD': cD_loss_coef}
    weight_dict['loss_giou'] = giou_loss_coef


    losses = ['labels', 'center_degree', 'cardinality']

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=eos_coef, losses=losses)
    criterion.to(device)
    return model, criterion


