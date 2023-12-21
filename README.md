# M<sup>3</sup>ONet: A Generalized Multi Modal Multi-Organ Network for Precise Segmentation of Medical Images

Authors: Snehashis Chakraborty, Komal Kumar, Abhijit Das, Balakrishna Reddy Pailla, Sudipta Roy

This repo contains the source code for our paper M<sup>3</sup>ONet: A Generalized Multi Modal Multi-Organ Network for Precise Segmentation of Medical Images.

#### Features

- M<sup>3</sup>ONet relies on a strong encoding pathway, enabling precise segmentation of target lesions across diverse regions
- The integration of a novel module named DMBC in M<sup>3</sup>ONet helps the model to focus on the intrinsic features for better understanding of the interclass relationship.
- M<sup>3</sup>ONet undergoes testing on five distinct multimodal datasets, showcasing its exceptional capacity for generalization compared to other state-of-the-art models.
- M<sup>3</sup>ONet also exhibit low computational complexity that makes it suitable for real world applications.

#### Usage

1. Install all the required packages from the requirment.txt file.
2. Create an object of M<sup>3</sup>ONet by the following code:
     ```
     m3onet = M3ONet.model(num_classes, output_activation, input_shape = (256,256,3))
     # provide num_classes, output_activation, and input_shape based on your  requirment.
     ```
3. Once the object is created, compile and train the model based on your requirement.

#### Dataset used in our paper

1. DUKE Breast Cancer dataset.
2. BraTS20202.
3. KiTS2023.
4. INBreast.
5. FracAtlas.

#### Result snippet on DUKE

| Input | Ground Truth | Output |
|:-----------:|:--------:|:------------:|
| ![Input Image](https://github.com/Snehashis100/M3ONet/blob/main/media/input_imgs.gif)| ![Ground Truth Image](https://github.com/Snehashis100/M3ONet/blob/main/media/gt_imgs.gif) | ![Output Image](https://github.com/Snehashis100/M3ONet/blob/main/media/output_imgs.gif) |

## Citation
If you found OneFormer useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

  ```bibtex

```
 
