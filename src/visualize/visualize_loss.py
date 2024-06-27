import matplotlib.pyplot as plt
import numpy as np
import cv2

from io import BytesIO
  

def generate_loss_plot(losses, save_path=None):

    # Plot the losses
    plt.plot(losses)
    plt.ylim(bottom=0)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Entwicklung')


    # Save the plot to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    image_bytes = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    img_cv2 = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    
    return img_cv2  