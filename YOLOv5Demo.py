#%%
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Image
im = 'star-wars-star-wars-battle-the-battle-wallpaper-preview.jpg'#'data/726.jpeg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(im)

results.pandas().xyxy[0]
# %%
