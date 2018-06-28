# Training of Convolutional Networks on Multiple Heterogeneous Datasets for Street Scene Semantic Segmentation (IV 2018)
Code for reproducing results for IV2018 paper "Training of Convolutional Networks on Multiple Heterogeneous Datasets for Street Scene Semantic Segmentation".

__Panagiotis Meletis and Gijs Dubbelman (2018)__ _Training of convolutional networks on multiple heterogeneous datasets for street scene semantic segmentation._ The 29th IEEE Intelligent Vehicles Symposiom (IV 2018), [full paper on arXiv](https://arxiv.org/abs/1803.05675).

If you find our work useful for your research, please cite the following paper:
```
@inproceedings{heterogeneous2018,
  title={Training of Convolutional Networks on Multiple Heterogeneous Datasets for Street Scene Semantic Segmentation},
  author={Panagiotis Meletis and Gijs Dubbelman},
  booktitle={2018 IEEE Intelligent Vehicles Symposium (IV)},
  year={2018}
}
```

# Code usage
See [here](hierarchical-semantic-segmentation/README.md).

# Paper summary
Discrimative power and generalization capabilities of convolutional networks is vital for deployment of semantic segmentation systems in the wild. These properties can be obtained by training a single net on multiple datasets.

Combined training on multiple datasets is hampered by a variety of reasons, mainly including:
* different level-of-detail of labels (e.g. *person* label in *dataset A* vs *pedestrian* and *rider* labels in *dataset B*)
* different annotation types (e.g. *per-pixel* annotations in *dataset A* vs *bounding box* annotations in *dataset B*)
* class imbalances between datasets (e.g. class person has 10^3 annotated pixels in *dataset A* and 10^6 pixels in *dataset B*)

We propose to construct a hierarchy of classifiers to combat above challenges. Hierarchical Semantic Segmentation is based on ResNet50. Its main novelty compared to other semantic segmentation systems, is that a single model can handle a variety of different datasets, with disjunct sets of semantic classes. Our system also runs in real time 18fps @512x1024 resolution. Figures 1-3 below provide sample results, from 3 different datasets.

![Image 1.1](sample_results/1/image_1.png "Image 1.1") | ![Image 1.2](sample_results/1/image_2.png "Image 1.2") | ![Image 1.3](sample_results/1/image_3.png "Image 1.3")
----|----|----
![Predictions 1.1](sample_results/1/predictions_1.png "Predictions 1.1") | ![Predictions 1.2](sample_results/1/predictions_2.png "Predictions 1.2") | ![Predictions 1.3](sample_results/1/predictions_3.png "Predictions 1.3")
![Ground truth 1.1](sample_results/1/ground_truth_1.png "Ground truth 1.1") | ![Ground truth 1.2](sample_results/1/ground_truth_2.png "Ground truth 1.2") | ![Ground truth 1.3](sample_results/1/ground_truth_3.png "Ground truth 1.3")

__Figure 1.__ Cityscapes validation split image examples - __top: input images, center: predictions, bottom: ground truth__. The network predictions include decisions from L1-L3 levels of the hierarchy. Note that the ground truth includes only one traffic sign superclass (yellow) and no road attribute
markings.

![Image 2.1](sample_results/2/image_1.jpg "Image 2.1") | ![Image 2.2](sample_results/2/image_2.jpg "Image 2.2") | ![Image 2.3](sample_results/2/image_3.jpg "Image 2.3")
----|----|----
![Predictions 2.1](sample_results/2/predictions_1.png "Predictions 2.1") | ![Predictions 2.2](sample_results/2/predictions_2.png "Predictions 2.2") | ![Predictions 2.3](sample_results/2/predictions_3.png "Predictions 2.3")
![Ground truth 2.1](sample_results/2/ground_truth_1.png "Ground truth 2.1") | ![Ground truth 2.2](sample_results/2/ground_truth_2.png "Ground truth 2.2") | ![Ground truth 2.3](sample_results/2/ground_truth_3.png "Ground truth 2.3")

__Figure 2.__ Mapillary Vistas validation split image examples - __top: input images, center: predictions, bottom: ground truth__. The network predictions include decisions from L1-L3 levels of the hierarchy. Note that the ground truth does not include traffic sign subclasses.

![Image 3.1](sample_results/3/image_1.png "Image 3.1") | ![Image 3.2](sample_results/3/image_2.png "Image 3.2") | ![Image 3.3](sample_results/3/image_3.png "Image 3.3")
----|----|----
![Predictions 3.1](sample_results/3/predictions_1.png "Predictions 3.1") | ![Predictions 3.2](sample_results/3/predictions_2.png "Predictions 3.2") | ![Predictions 3.3](sample_results/3/predictions_3.png "Predictions 3.3")
![Ground truth 3.1](sample_results/3/ground_truth_1.png "Ground truth 3.1") | ![Ground truth 3.2](sample_results/3/ground_truth_2.png "Ground truth 3.2") | ![Ground truth 3.3](sample_results/3/ground_truth_3.png "Ground truth 3.3")

__Figure 3.__ TSDB test split image examples - __top: input images, center: predictions, bottom: ground truth__. The network predictions include decisions  from  L1-L3  levels  of  the  hierarchy.  Note  that  the  ground  truth includes only traffic sign bounding boxes, since rest pixels are unlabeled.

