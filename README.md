KAHAWAI (Streamflow) Project

- index.html: project web page (unused)
- images: images for the web page
- pixel counting: experiments with water segmentation and pixel counting
- water_stage_classification: all training code for resnet50 embedding classifier
- utilities:
    - tls_frame_extractor: notebook to extract images from .tls timelapse videos
    - data_sheet_creator: makes an excel sheet with all image paths and timestamps
    - missing: usgs_data_sheet_creator (to get usgs data on a sheet), data_fuser (to combine paths with usgs data)
 
Data Sheets can be found in this drive:
https://drive.google.com/drive/folders/1mXRH7iA6P0N-J982yP7fZhFcnFB25WYP?ths=true



Workflow:
- collect image data from cameras
- call tls_frame_extractor to get the individual images
- call data_sheet_creator to put image paths and timestamps into a sheet
- get usgs data and put it on the datasheet
- add labels to the sheet with add_labels
- upload sheet and images to koa
- clone repo into koa
- get_feature_dataset to create dataset of images and save that on koa
- train1
