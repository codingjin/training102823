import os
import torch
from PIL import Image

folder_path = "/home/jin/workspace/luadsc3125/wsi_data/luad_0.3125"
filenames = os.listdir(folder_path)

for filename in filenames:
    filepath = '/'.join([folder_path, filename])
    #print(filepath)
    image = Image.open(filepath)
    width, height = image.size
    print(f"{width}\t{height}")

