# object detection and identification (1)

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import cv2


# url = "/content/download (1).jpg"
# image = Image.open(url)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

def identifier(image, image_name):
    image = Image.open(image)

# you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    width, height = image.size # use image.size to get width and height
    target_sizes = torch.tensor([[height, width]]) # Create a tensor with the height and width of the image
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    d ={}
    detection_results = "" 


    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        d[model.config.id2label[label.item()]] = box
        detection_results += (
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} \n"
        )

    # colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
    #       (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Loop through the coordinates and draw rectangles with different colors
    # for i,(k , (x, y, w, h)) in enumerate(d.items()):
    #     # Convert coordinates to integers
    #     x, y, w, h = int(x), int(y), int(w), int(h)
    #     color = colors[i % len(colors)]  # Cycle through colors
    #     cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
 
    # cv2.imwrite(f"C:\\Users\\vishal sharma\\Desktop\\project_root\\data\\output\\detected_{image_name}", image)
    # return f"C:\\Users\\vishal sharma\\Desktop\\project_root\\data\\output\\detected_{image_name}"
    return d , detection_results












