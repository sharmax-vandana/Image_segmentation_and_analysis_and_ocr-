import cv2
def postp(d,img):
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i,(k , (x, y, w, h)) in enumerate(d.items()):
    # Convert coordinates to integers
        x, y, w, h = int(x), int(y), int(w), int(h)
        color = colors[i % len(colors)]  # Cycle through colors
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return img