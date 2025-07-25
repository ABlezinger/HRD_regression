import hashlib
import sys

def compute_md5(file_path, chunk_size=8192):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

files = [
"1a1a480e-e6a5-46fd-ac72-e9e38d792c2f/TCGA-AA-3527-01A-01-BS1.5aa0bd9a-172a-4109-81e2-5293fba47e7c.svs",
"3862a74b-3485-44b6-a67e-1a5d723d18fe/TCGA-AA-3688-01A-01-TS1.90615e6d-6dbf-4d4e-aea5-21ec1bbb67ed.svs",
"69b4eafa-a83c-4e5a-8855-95de51585362/TCGA-AA-3688-01Z-00-DX1.642ce194-6dc0-4a96-aa79-674f48966df3.svs",
"885ee2ab-38a2-43a3-b963-8861763c5bd3/TCGA-AA-3527-01A-01-TS1.612d33b3-569d-4ea0-9dd2-f08a1ba61707.svs",
"c870e832-73a7-4b5a-b1bd-c7aabb10dbf5/TCGA-AA-3672-01A-01-BS1.a0ceb6c5-20ba-4a0b-a957-2079ee83adb9.svs"]
for file in files:
    checksum = compute_md5(f"test_WSI_test/{file}")
    print(f"MD5 checksum: {checksum}")
    print("File: ", file)