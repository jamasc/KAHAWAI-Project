KAHAWAI (Streamflow) Project

- data
    all code to create datasheets with image ID's and timestamps
- models
    all CV code
    - pixel counting: experiments with water segmentation and pixel counting
    - water_stage_classification: all training code for resnet50 embedding classifier
- website (unused)
    simple webpage and images,



Data Sheets can be found in this drive:
https://drive.google.com/drive/folders/1mXRH7iA6P0N-J982yP7fZhFcnFB25WYP?ths=true



Workflow for deployed cams:
- collect image data from cameras
- call tls_frame_extractor to get the individual images
- call data_sheet_creator to put image paths and timestamps into a sheet
- get usgs data and put it on the datasheet
- add labels to the sheet with add_labels
- upload sheet and images to koa
- clone repo into koa
- get_feature_dataset to create dataset of image features and save that on koa
- train1

Workflow for DAR dataset:
- upload to KOA
- get embeddings
- train
