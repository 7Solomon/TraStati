
import configure

import cv2


def visualize_image(image, title="Image"):
    if configure.display_mode == "cv2":
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif configure.display_mode == "plt":
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.show()