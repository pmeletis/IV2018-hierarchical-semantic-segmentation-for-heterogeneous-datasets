In this directory you can find the following three datasets used in the paper:
1. **Cityscapes Extended (v1)**: We extended the 33 original Cityscapes labels with 34 traffic sign classes from the GTSDB dataset, into a fully compatible combined label set.
   
   Since, Cityscapes annotations for traffic signs are not the instance, traffic signs that are touch each other in the image need to be separated. This is done automatically by a simple binary mask segmentation algorithm, which can be found [here](https://github.com/pmeletis/IV2018-hierarchical-semantic-segmentation-for-heterogeneous-datasets/edit/master/datasets). This procedure is not perfect, thus expect few wrong annotations (that we plan to fix in v2 of the dataset). It would be great to send us any wrong or missing annotation you spot.
   
2. **GTSDB per pixel (coarse)**: transformed bounding box ground truth into coarse per pixel annotations, i.e. every pixel inside the bounding box is assigned the class of the bounding box.
3. **GTSDB per pixel (fine)**: transformed bounding box ground truth into fine per pixel annotations, i.e. only traffic sign pixels are assigned the class of the associated bounding box.
