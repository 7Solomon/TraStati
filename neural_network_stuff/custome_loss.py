import torch

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment


import math

def calculate_similarity(pred_box, target_box):
    pred_x, pred_y, pred_degree = pred_box
    target_x, target_y, target_degree = target_box

    spatial_distance = math.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2)
    angular_difference = abs((pred_degree - target_degree + 180) % 360 - 180)

    similarity = 1 / (1 + spatial_distance) + 1 / (1 + angular_difference)
    return similarity


class CustomeDataLoss(torch.nn.Module):
    def __init__(self):
        super(CustomeDataLoss, self).__init__()
        self.degree_weight = 0.2
        self.x_weight = 0.4
        self.y_weight = 0.4

    def forward(self, predicted, target):
        assert predicted.shape == target.shape, "Shapes of predicted and target must be the same"
       
        predicted_x, predicted_y, predicted_degree = predicted.split(1, dim=2)
        target_x, target_y, target_degree = target.split(1, dim=2)

        print(f'Predicted X:{predicted_x} Predicted Y:{predicted_y} Predicted Degree:{predicted_degree}')
        print(f'Target X:{target_x} Target Y:{target_y} Target Degree:{target_degree}')

        loss_x = torch.mean((predicted_x - target_x)**2)
        loss_y = torch.mean((predicted_y - target_y)**2)

        loss_degree = torch.mean(((predicted_degree - target_degree) % 360)**2)
        total_loss = loss_x*self.x_weight + loss_y*self.y_weight + loss_degree*self.degree_weight
        print(f'Data Loss:{total_loss}')
        return total_loss
    
class CustomeClassLoss(torch.nn.Module):
    def __init__(self):
        super(CustomeClassLoss, self).__init__()

    def forward(self, predicted_logits, target_labels):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(predicted_logits, target_labels)
        print(f'Class Loss:{loss}')
        return loss
    

class CustomeComLoss(torch.nn.Module):
    def __init__(self):
        super(CustomeComLoss, self).__init__()
        self.class_criterion = CustomeClassLoss()
        self.data_criterion = CustomeDataLoss()
        self.class_weight = 0.9
        self.data_weight = 0.1

    def get_order(self, predicted,target):
        assert predicted.shape[0] == target.shape[0] , ' predicted.shape[0] != target.shape[0], at graph stuff'
        for b in range(predicted.shape[0]):
            G = nx.Graph()
            G.add_nodes_from([(i, {'type': 'prediction', 'box': bbox}) for i, bbox in enumerate(predicted[b])], bipartite=0)
            G.add_nodes_from([(j, {'type': 'target', 'box': bbox}) for j, bbox in enumerate(target[b])], bipartite=1)

            
            edge_weights = np.zeros((len(predicted[b]), len(target[b])))

            for i, pred_box in enumerate(predicted[b]):
                for j, target_box in enumerate(target[b]):
                    edge_weights[i, j] = calculate_similarity(pred_box, target_box)


            row_ind, col_ind = linear_sum_assignment(-edge_weights)  # Note the negative sign for maximization

            # Print the optimal matching
            double_target = target[b][:]
            for i, j in zip(row_ind, col_ind):
                #pred_node = (i, {'type': 'prediction', 'box': predicted[b][i]})
                #target_node = (j, {'type': 'target', 'box': target[b][j]})
                #print(f"Prediction {i} -> Target {j}")
                double_target[i] = target[b][j]
            target[b] = double_target
        return predicted,target


    def forward(self,predicted,target):
        # Hinzufügen der prob
        target_classes = target['classes']
        target['classes'] = torch.eye(6)[target_classes]   # For data   6 = num classes

        predicted['data'],target['data'] = self.get_order(predicted['data'],target['data'])

        class_loss = self.class_criterion(predicted['classes'], target['classes'])
        data_loss = self.data_criterion(predicted['data'], target['data'])
        combined_loss = class_loss*self.class_weight + data_loss*self.data_weight
        return combined_loss