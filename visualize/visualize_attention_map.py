import cv2
import numpy as np

def draw_cachel(heatmaps):
    width, height = heatmaps[0].shape[:2]
    # BruteForce for 4
    top = np.hstack(heatmaps[:2])
    bot = np.hstack(heatmaps[2:])

    display_heatmaps = np.vstack([top,bot])
    return cv2.resize(display_heatmaps, (height, width))


    """ 
    length = len(heatmaps)
    if length%2 == 0:
        top = np.hstack(heatmaps[length//2:])
        bot = np.hstack(heatmaps[:length//2])

       
        display_heatmaps = np.vstack([top,bot])
        display_heatmaps = cv2.resize(display_heatmaps, heatmaps[0].shape[:2])
        return display_heatmaps
    else:
        top = np.hstack(heatmaps[:length//2])
        bot = np.hstack(heatmaps[length//2:])
    

        display_heatmaps = np.vstack([top,bot,heatmaps[-1]])
        display_heatmaps = cv2.resize(display_heatmaps, heatmaps[0].shape[:2])
        return display_heatmaps
    """

def attention_map(attention_weights, original_image):

    heatmaps = []
    for layer in attention_weights:
        for weigths_per_head in layer: 
            attention_map = weigths_per_head.squeeze().detach().numpy()
            attention_map = (attention_map * 255).astype(np.uint8)  # Convert to uint8 for visualization

            resized_attention_map = cv2.resize(attention_map, (original_image.shape[1], original_image.shape[0]))
            heatmap = cv2.applyColorMap(resized_attention_map, cv2.COLORMAP_HOT)


            heatmaps.append(heatmap)
    
    
    
    for i, heatmap in enumerate(heatmaps):
        display_heatmaps = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
        heatmaps[i] = display_heatmaps
    
    display_heatmaps = draw_cachel(heatmaps)
    

    cv2.imshow('Attention Stuff', display_heatmaps)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
