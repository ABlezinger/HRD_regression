import os
import json
import shutil
from glob import glob


PATH = "/data/datasets/images/TCGA/WSIs"
# Get all files recursively
all_files = glob(os.path.join(PATH, "**"), recursive=True)
# Filter to only files (not directories)
svs_list = [f for f in all_files if os.path.isfile(f)]
print(svs_list[:5])
file_names = [f.split("/")[-1] for f in svs_list]

tcga = json.load( open("datafiles/TCGA_complete.json"))
info = dict()

for image in tcga:
    if image["file_name"] in file_names:
        cohort = image["cases"][0]["project"]["project_id"].split("-")[-1]
        info[cohort] = info.get(cohort, 0) + 1
        if cohort == "BRCA":
            os.makedirs(f"/data/datasets/images/TCGA/{cohort}", exist_ok=True)
            shutil.move(f"{PATH}/{image['file_id']}/{image['file_name']}", f"/data/datasets/images/TCGA/{cohort}" )
    else: 
        pass
    
print("moved files:")
print(info)
        
        

