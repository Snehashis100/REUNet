# Robust Semantic Learning for Precise Medical Image Segmentation

Authors: Snehashis Chakraborty, Komal Kumar, Ankan Deria, Dwarikanath Mahapatra, Behzad Bozorgtabar, Sudipta Roy

This repo contains the source code for our paper Robust Semantic Learning for Precise Medical Image Segmentation.

<p align="center">
  <img src="https://github.com/Snehashis100/M3ONet/assets/63040034/3c668bfa-f470-46ad-bf98-ef46777f9b2f" alt="Image Description">
</p>

## Features

- REUnet relies on a strong encoding pathway, enabling precise segmentation of target lesions across diverse regions
- The integration of a novel module named DMBC in REUnet helps the model to focus on the intrinsic features for better understanding of the interclass relationship.
- REUnet undergoes testing on five distinct multimodal datasets, showcasing its exceptional capacity for generalization compared to other state-of-the-art models.
- REUnet also exhibit low computational complexity that makes it suitable for real world applications.

## Usage

1. Install all the required packages from the requirment.txt file.
2. Create an object of REUnet by the following code:
     ```
     reunet = REUnet.model(num_classes, output_activation, input_shape = (256,256,3))
     # provide num_classes, output_activation, and input_shape based on your  requirment.
     ```
3. Once the object is created, compile and train the model based on your requirement.

## Dataset used in our paper

1. DUKE Breast Cancer dataset.
2. BraTS20202.
3. KiTS2023.
4. INBreast.
5. FracAtlas.

## Result snippet on DUKE

| Input | Ground Truth | Output |
|:-----------:|:--------:|:------------:|
| ![Input Image](https://github.com/Snehashis100/M3ONet/blob/main/media/input_imgs.gif)| ![Ground Truth Image](https://github.com/Snehashis100/M3ONet/blob/main/media/gt_imgs.gif) | ![Output Image](https://github.com/Snehashis100/M3ONet/blob/main/media/output_imgs.gif) |

## Some more results
<p align="center">
  <img src="https://github.com/Snehashis100/M3ONet/assets/63040034/05eb92c9-4e51-4b0f-8f95-e4d054c84227" alt="Image Description">
</p>

## Citation
If you found OneFormer useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

  ```bibtex
will be provided once published
```
 
