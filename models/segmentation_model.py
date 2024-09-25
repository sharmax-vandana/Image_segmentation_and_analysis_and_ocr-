# image segmentation  1.0

from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
import cv2 
import matplotlib.pyplot as plt
import numpy as np

# load MaskFormer fine-tuned on COCO panoptic segmentation
processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")

def segm(image,image_name):
    image = Image.open(f"{image}")
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    # model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    # you can pass them to processor for postprocessing
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
    predicted_panoptic_map = result["segmentation"]


# result['segments_info']


# printing the segmented image ------------------------------------------------------------------------------------------------------------




# Assuming 'image_tensor' is your image tensor
# image_tensor = ...  # shape: (height, width, channels)

# Convert tensor to numpy array and clip values to [0, 255]
    image_array = np.clip(predicted_panoptic_map.numpy() , 0, 255).astype(np.uint8) # convert the tensor to a numpy array using .numpy()

# Display image
    plt.imshow(image_array)
    plt.axis('off')  # Hide axes
    # plt.show()
    plt.savefig(f"..\\data\\output\\{image_name}",)
    return f"..\\data\\output\\{image_name}"