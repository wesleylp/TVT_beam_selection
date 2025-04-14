# Customized image front end
Note: Please keep the directory structure as indicated.
Note: since feature generation takes time we recommend using bash mode for running.

----------------------------------------------
## If you want to USE feature extractor
run this command:

```bash
python main.py \ 
    --base_path <base_path_to_code_folder> \
    --train False \
    --test False \
    --restore_models True \
    --model_path <where_models_are_saved_loaded> \
    --model_json <path_to_json> \
    --model_weight <path_to_hdf_file> \
    --path_of_entire_image <path_to_the_source_images>
```

---------------------------------------------------
## If you want to CREATE the feature extractor from scratch


1. You need to create a dataset of 40x40 samples of car, truck, bus, background, in order to do that.
    a) some samples from car, truck, bus, background are available in "samples" folder.
    b) You need to crop these samples with window size of 40x40 and stride size of 5 (We drived these numbers empricaly.). For that you can use crop.py.

2. Balance the dataset.
The generated crops are not balanced. So we try to balance it by using data augmentation and adding light effects. you can use uniform.py script for that.

3. Split dataset.
We split the dataset to train/validation/test using split.py.

4. Run this command:
```bash
python main.py\
    --base_path <base_path_to_code_folder> \
    --train True \ 
    --test True  \
    --restore_models False \
    --model_path <where_models_are_saved_loaded> \  
    --path_of_entire_image <path_to_the_source_images>
```

---------------------------------------------------

## After generating features:
We extract the type of receiver from `CoordVehiclesRxPerScene_s009.csv` and discard non-relevant images. For that you can use pipeline.py.

