import torch
from CLIP import clip
from PIL import Image, ImageDraw
import cv2

### Detect BBoxes

# Models
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
img = 'woof_meow.jpg'

# Query
q = "A photo of a cat"

# Inference
results = yolo_model(img)

# Results
results.print()
results.show()
#      xmin    ymin    xmax   ymax  confidence  class    name


### Choose best BBox

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(q).to(device)

max_prob = 0
for bbox in results.xyxy[0].cpu().numpy():
    image = cv2.imread(img)
    image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    cv2.imwrite("crop.png", image)
    image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        prob = logits_per_text.cpu().numpy()[0][0]
        print(prob)
        if prob > max_prob:
            max_prob = prob
            best_bbox = bbox

source_img = Image.open(img).convert("RGBA")
draw = ImageDraw.Draw(source_img)
draw.rectangle(((best_bbox[0], best_bbox[1]), (best_bbox[2], best_bbox[3])))
source_img.save("test.png", "PNG")