In this directory you can find the following three datasets used in the paper:
1. **Cityscapes Extended (v1)**: We extended the 33 original Cityscapes labels with 43 traffic sign classes from the GTSDB dataset, into a fully compatible combined label set.
   
   Since, Cityscapes annotations for traffic signs are not per instance, traffic signs that are touching each other in the image need to be separated. This is done automatically by a simple binary mask segmentation algorithm, which can be found [here](https://github.com/pmeletis/IV2018-hierarchical-semantic-segmentation-for-heterogeneous-datasets/master/datasets). This procedure is not perfect, thus expect few wrong annotations (that we plan to fix in v2 of the dataset). It would be great to send us any wrong or missing annotation you spot.
   
   The selected label ids for the traffic sign classes should follow the Cityscapes scheme and enable future extensions of the Cityscapes 33 ids. For those reasons we assign traffic signs ids from 2000 to 2043 (100 * 20 (traffic sign id)). The combined labels file can be found [here](https://github.com/pmeletis/IV2018-hierarchical-semantic-segmentation-for-heterogeneous-datasets/blob/master/datasets/labels.py). As a consequence, the ground truth images are of *int32* dtype, unlike original Cityscapes' *uint8*, and can be verified in Python:
   ```python
   from PIL import Image
   import numpy as np
   label_pil = Image.open(<label filepath>)
   label_pil
   label_np = np.array(label_pil)
   label_np.dtype
   ```
2. **GTSDB per pixel (coarse)**: transformed bounding box ground truth into coarse per pixel annotations, i.e. every pixel inside the bounding box is assigned the class of the bounding box.
3. **GTSDB per pixel (fine)**: transformed bounding box ground truth into fine per pixel annotations, i.e. only traffic sign pixels are assigned the class of the associated bounding box.
