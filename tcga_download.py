import json
import requests
import re
import tarfile
from tqdm import tqdm 
import os 

# Download Params
DATASET_PATH = "/data/datasets/images/TCGA"
data_endpt = "https://api.gdc.cancer.gov/data"

BATCH_SIZE = 5
# BATCH_SIZE = 2

print(f"Writing images to {DATASET_PATH}")

# loading progess files 
files = json.load(open(f"datafiles/TCGA_02.json"))

uuids = [case["file_id"] for case in files]

if os.path.exists("datafiles/failed_uuids.json"):
    failed_uuids = json.load(open("datafiles/failed_uuids.json"))
else:
    failed_uuids = []


if os.path.exists("datafiles/download_progress_2.json"):
    download_progress = json.load(open("datafiles/download_progress_2.json"))
else:
    download_progress = {"last_successful": 0}

print(f"Number of files: {len(uuids)}")
batches = len(uuids) // BATCH_SIZE + 1

batch = 0

for i in tqdm(range(0, len(uuids), BATCH_SIZE), total=batches):
    
    #skipping done batches
    if download_progress["last_successful"] > batch:
        print(f"Skipping batch {batch} as it was already downloaded.")
        batch += 1
        continue
    
    #request data
    params = {"ids": uuids[i:i+BATCH_SIZE]}
    response = requests.post(data_endpt,
                            data = json.dumps(params),
                            headers={
                                "Content-Type": "application/json"
                                })

    
    #handle response
    response_head_cd = response.headers["Content-Disposition"]
    # print("KEYS")
    # print(response.headers.keys())

    file_name = re.findall("filename=(.+)", response_head_cd)[0]

    
    filepath = f"{DATASET_PATH}/{file_name}"

    with open(filepath, "wb") as output_file:
        output_file.write(response.content)
    
    #check for broken tar file 
    try:    
        file = tarfile.open(filepath)
        # print(file.getmembers())
    except EOFError as e:
        file.close()
        os.remove(filepath)
        print(f"removed file {filepath} due to EOFError:")
        failed_uuids += params["ids"]
    else:
        file.close()
        print(f"Batch {batch} worked")
    
    download_progress["last_successful"] = batch
    json.dump(download_progress, open("datafiles/download_progress_2.json", "w"))
    batch += 1
    
json.dump(failed_uuids, open("datafiles/failed_uuids.json", "w"))
print(f"downloaded {len(uuids) - len(failed_uuids)} out of {len(uuids)} ({((len(uuids) - len(failed_uuids))/len(uuids))*100}%) len files that failed to download.")