import os

folder_path = "/home/jin/workspace/luadsc125/wsi_data1.25/wsi_data_1.25/luad_1.25"

filenames = os.listdir(folder_path)

for filename in filenames:
    print(f"{filename}\t0")
    #print(f"{filename}")


