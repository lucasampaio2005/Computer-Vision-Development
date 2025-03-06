import torch
from torchvision import models, transforms
from PIL import Image
import cv2

model = models.resnet50(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_path = 'data/2.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)
image_tensor = preprocess(image_pil).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    _, predicted = output.max(1)
    scene_label = predicted.item()

    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]
        scene_description = labels[scene_label]

        cv2.putText(image, scene_description, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Scene Classification', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()