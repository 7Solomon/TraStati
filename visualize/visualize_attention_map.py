import cv2
import numpy as np


def attention_map(attention_weights, original_image):
    print(original_image.shape)

    heatmaps = []
    for layer in attention_weights:
        for weigths_per_head in layer: 
            attention_map = weigths_per_head.squeeze().detach().numpy()
            attention_map = (attention_map * 255).astype(np.uint8)  # Convert to uint8 for visualization

            resized_attention_map = cv2.resize(attention_map, (480, 480))
            heatmap = cv2.applyColorMap(resized_attention_map, cv2.COLORMAP_HOT)
            heatmaps.append(heatmap)
    display_heatmaps = np.hstack([*heatmaps, original_image])

    cv2.imshow('Attention Stuff', display_heatmaps)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
