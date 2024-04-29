
import configure

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image



def visualize_image(image, title="Image", save_path=None):
    if configure.display_mode == "cv2":
        if save_path is not None:
            cv2.imwrite(save_path, image)
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif configure.display_mode == "plt":
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    elif configure.display_mode == "pil":
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img.show()
    else:
        raise ValueError("Invalid display mode. Please choose 'cv2', 'plt' or 'pil'.")