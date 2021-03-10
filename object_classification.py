import torch
import torchvision
from PIL import Image
import json
# class_idx = json.load(open("imagenet_class_index.json"))

I = Image.open('dog.jpg')
model = torchvision.models.resnext101_32x8d(pretrained=True, progress=False)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Scale(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

model.eval()
p = model(transform(I)[None])[0]
print(p)
# print( ' , '.join([class_idx[str(int(i))][1] for i in p.argsort(descending=True)[:5]]) )

