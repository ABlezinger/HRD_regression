import json
import requests
import re
import tarfile
from tqdm import tqdm 
import os 

# Download Params
DATASET_PATH = "/data/datasets/images/CPTAC"
data_endpt = "https://api.gdc.cancer.gov/data"

BATCH_SIZE = 1
# BATCH_SIZE = 2

print(f"Writing images to {DATASET_PATH}")

# loading progess files 
files = json.load(open(f"datafiles/TCGA_complete.json"))

cases = [case for case in files if case["cases"][0]["project"]["project_id"] in ["TCGA-BRCA"]]
# uuids = [case["file_id"] for case in files if case["cases"][0]["project"]["project_id"] in ["TCGA-BRCA"]]
brca_list = os.listdir(f"{DATASET_PATH}/BRCA")
uuids = [case["file_id"] for case in cases if f"{case['file_name']}" not in brca_list]

if os.path.exists("datafiles/failed_uuids_test.json"):
    failed_uuids = json.load(open("datafiles/failed_uuids_test.json"))
else:
    failed_uuids = []


# if os.path.exists("datafiles/download_progress.json"):
#     download_progress = json.load(open("datafiles/download_progress.json"))
# else:
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
    try:
        response = requests.post(data_endpt,
                                data = json.dumps(params),
                                headers={
                                    "Content-Type": "application/json"
                                    }, timeout=600)
    except:
        print(f"retrying batch {batch} due to error:")
        response = requests.post(data_endpt,
                                data = json.dumps(params),
                                headers={
                                    "Content-Type": "application/json"
                                    }, timeout=600)
    
    #handle response
    try:
        response_head_cd = response.headers["Content-Disposition"]
    
    except KeyError as e:
        print(f"batch {batch} failed: KeyError")
        failed_uuids += params["ids"]
        batch += 1
        continue
    file_name = re.findall("filename=(.+)", response_head_cd)[0]

    
    filepath = f"{DATASET_PATH}/BRCA/{file_name}"

    with open(filepath, "wb") as output_file:
        output_file.write(response.content)
    
    #check for broken tar file 
    # try:    
    #     file = tarfile.open(filepath)
    #     members = file.getmembers()
    # except EOFError as e:
    #     file.close()
    #     # os.remove(filepath)
    #     print(f"{filepath} EOFError:")
    #     failed_uuids += params["ids"]
    # else:
    #     file.close()
    #     print(f"Batch {batch} worked")
    
    # download_progress["last_successful"] = batch
    # json.dump(download_progress, open("datafiles/download_progress.json", "w"))
    # batch += 1
    
json.dump(failed_uuids, open("datafiles/failed_uuids_test.json", "w"))
print(f"downloaded {len(uuids) - len(failed_uuids)} out of {len(uuids)} ({((len(uuids) - len(failed_uuids))/len(uuids))*100}%) len files that failed to download.")