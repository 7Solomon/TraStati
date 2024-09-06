# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

#from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from src import configure

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_cD: float = 1):
        """Creates the matcher

        Params: 
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_cd = cost_cD

        assert cost_class != 0 or cost_cD != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):

        bs, num_queries = outputs["outputs_class"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["outputs_class"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_cds = outputs["output_center_degree_points"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["classes"] for v in targets])
        tgt_cds = torch.cat([v["data"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_cd = torch.cdist(out_cds, tgt_cds, p=1)

        # Final cost matrix
        C = self.cost_cd * cost_cd + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["data"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher():
    return HungarianMatcher(cost_class=configure.cost_class, cost_cD=configure.cost_cD)
