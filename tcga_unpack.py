import tarfile
import glob
import os

WSI_DIR = "/data/datasets/images/TCGA"


unpack = True

# file_list = glob.glob("/data/datasets/images/TCGA/*.tar.gz")
# print("LEN: " + str(len(file_list)))
# print(file_list[:5])

# file = file_list[143]

# print(f"unpacking... {file}")
# file = tarfile.open(file)
# mems = file.getmembers()
# print(mems)
# # file.extractall(WSI_DIR)
# # file.close()

# file_list = glob.glob("/data/datasets/images/TCGA/*.tar.gz")
# print("LEN: " + str(len(file_list)))
# print(file_list[200])

if unpack:
    # file = tarfile.open(f"{test_WSI_test}/gdc_download_20250722_173536.403636.tar.gz")
    file = tarfile.open(f"{WSI_DIR}/gdc_download_20250722_190801.401769.tar.gz")
    print(file.getmembers())
    # file.extractall("test_WSI_test")
    file.close()