import json

import numpy as np
import pandas as pd
from pathlib import Path


# tcga_list = json.load(open("datafiles/TCGA_01.json")) + json.load(open("datafiles/TCGA_02.json"))
tcga_list = json.load(open("datafiles/TCGA_complete.json"))
patient_ids = ["-".join(file["file_name"].split("-")[:3]) for file in tcga_list]
cohorts = [file["cases"][0]["project"]["project_id"] for file in tcga_list]
feature_files = [f"{file['file_name'][:-4]}.h5" for file in tcga_list]

data_pd = pd.DataFrame({
    "patient_id": patient_ids,
    "cohort": cohorts,
    "feature_file": feature_files,
    "dataset": "TCGA"
})

hrd_values = pd.read_excel("datafiles/TCGA_CPTAC_HRD.xlsx").rename(
    columns={"Patient ID": "patient_id"})

print(hrd_values.head())

data_pd = data_pd.join(hrd_values.set_index("patient_id")[["HRD_sum", "HRD_Binary"]], on="patient_id", how="outer")

print(data_pd.sort_index().head())
print(data_pd.groupby("cohort").value_counts(["HRD_Binary"], dropna=False))

data_pd = data_pd.dropna(subset=["HRD_sum"])
data_pd = data_pd.dropna(subset=["patient_id"])
data_pd = data_pd.dropna(subset=["cohort"])

data_pd.to_excel("datafiles/TCGA_CPTAC_data.xlsx", index=False)
