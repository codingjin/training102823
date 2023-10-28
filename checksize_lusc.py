import os
import torch
from PIL import Image

folder_path = "/uufs/chpc.utah.edu/common/home/u1498392/wsi_data/lusc_0.3125"
filenames = os.listdir(folder_path)

for filename in filenames:
    filepath = '/'.join([folder_path, filename])
    #print(filepath)
    image = Image.open(filepath)
    width, height = image.size
    print(f"{width}\t{height}")

