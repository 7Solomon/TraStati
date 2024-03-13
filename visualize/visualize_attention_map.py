import torch
import matplotlib.pyplot as plt


def attention_map(attention_weights):
    layer_num = 0  # Specify which layer's attention map you want to visualize
    head_num = 0   # Specify which attention head within the layer you want to visualize

    attention_map = attention_weights[layer_num][head_num].squeeze().cpu().numpy()
    plt.imshow(attention_map, cmap='hot', interpolation='nearest')
    plt.title(f'Layer {layer_num} Head {head_num} Attention Map')
    plt.colorbar()
    plt.show()
