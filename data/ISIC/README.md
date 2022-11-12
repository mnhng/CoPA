## How to prepare ISIC data

1. Use [isic-cli](https://github.com/ImageMarkup/isic-cli) to download the ISIC images and metadata
```
    isic image download raw/
```
2. Merge the metadata in the csv file downloaded in Step 1 with the metadata in *metadata_size.csv*. The result should look like *metadata_columns.csv*.
3. Crop and resize the images by running 
```
    python preproc.py
```
