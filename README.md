# Get models for feature extraction

## For GPFM and RetCCL
download gpfm model: https://github.com/birkhoffkiki/GPFM/blob/master/models/dinov2/__init__.py

download RetCCL model: https://github.com/Xiyue-Wang/RetCCL

save the downloaded weights in the respctive directory (e.g. custom_WSI_pipeline/models/RetCCL/RetCCL.pth), otherwise adapt the checkpoint-path in the load_*modelname*() function

## For UNI, UNI_2, CONCH

create hugginface_config.json file with the HRD_Regression Package level

```
{
    "token": "YOUR_HUGGINFACE_TOKEN"
}
```

to download UNI, UNI_2, CONCH

for uni models application form on github needs to be submitted

# Code sources

## Marugoto WSI Pipeline:

The codebase for the WSI pipeline referenced in this paper by KatherLab (https://www.nature.com/articles/s41467-024-45589-1)  (available under https://github.com/KatherLab/end2end-WSI-preprocessing/releases/tag/v1.0.0-preprocessing) has been adapted for extracting features with different extraction models.
Additionally the code for loading each model has been copied and slightly adapted from each respective model github/hugginface documentation.


# Datasets
## TCGA
The Cohorts for the TCGA dataset can be downloaded with tcga_dowlnload.sh. If there occurs an error during the download the task can be restarted. The download_progress files in datafiles ensures that progress is noted and the download continues after restart
# todo
Datafiles combinen, sodass nur ein Task gestartet werden muss

## ASCLI required

# Python env for HRD_Pred Marugoto

```
conda create -n hrd_pred python=3.10
pip install -r hrd_prediction/requirements.txt
```