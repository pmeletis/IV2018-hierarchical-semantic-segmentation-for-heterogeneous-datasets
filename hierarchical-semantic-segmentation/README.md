# Requirements
Python 3.6, tensorflow, numpy, Pillow, scipy, matplotlib (only for inference), opencv-python (only for training)

Tested on Ubuntu 16.04 64bit, 64 GB RAM, 20 CPUs, Titan V 12 GB GPU with
matplotlib==2.2.2,
numpy==1.14.2,
opencv-python==3.4.0.12,
Pillow==5.1.0,
scipy==1.0.1,
tensorboard==1.6.0,
tensorflow==1.6.0.

# Inference

Prerequisite: due to github space limitations download the checkpoints from [here](https://www.dropbox.com/s/ia8sh80ai9awyqw/trained_model_ckpts.zip?dl=0) and unzip them into checkpoints directory.

To run inference on images on the combined label set of 108 classes from Cityscapes, Vistas and GTSDB datasets:
```python
python predict.py log_dir predict_dir [optional arguments]
```

Required arguments:
* log_dir: the checkpoints directory, the system will use the latest checkpoint in this directory
* predict_dir: a directory containing images (RGB, formats: png, jpg, jpeg, ppm). The system will parse the directory for all supported images and will do sequential prediction on them.

Optional arguments:
* --plotting: live plotting
* --export_color_images: exports color label images at a specific directory (requires --results_dir to be provided)
* --results_dir \<dir>: provide this directory when exporting flags ar eprovided
* --restore_emas: restore exponential moving averages instead of normal variables from the checkpoint (better results)

more arguments can be found at utils/utils.py in function add_predict_arguments.

Example:
```python
python predict.py checkpoints/trained_model_ckpts/training02 samples --restore_emas --plotting --export_color_images --results_dir samples/results
```

# Coming soon...
* exporting color images per dataset

# Base repository
Developed as an extension of the semantic segmentation system of this [repo](https://github.com/pmeletis/semantic-segmentation).
