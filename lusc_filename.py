import os

#folder_path = "/uufs/chpc.utah.edu/common/home/u1498392/wsi_data/lusc_0.3125"
folder_path = "/home/jin/workspace/luadsc3125/wsi_data/lusc_0.3125"

filenames = os.listdir(folder_path)

for filename in filenames:
    #print(f"{filename}\t1")
	print(filename)


