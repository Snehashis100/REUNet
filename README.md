# Robust Semantic Learning for Precise Medical Image Segmentation

Authors: Snehashis Chakraborty, Komal Kumar, Ankan Deria, Dwarikanath Mahapatra, Behzad Bozorgtabar, Sudipta Roy

This repo contains the source code for our paper Robust Semantic Learning for Precise Medical Image Segmentation which is accepted in Biomedical Signal Processing and Control, Elsevier.
<figure>
  <p align="center">
  <img src="https://github.com/user-attachments/assets/776c6560-d427-4593-830c-d85523394eb6" alt="Image Description" width="700" height="600">
</p>
  <figcaption>The architecture of REUnet with encoding and decoding route. Within the encoding route, each block (EB) is
composed of a sequence of DMBC modules. Each DMBC module (DMBCi) is accompanied by its respective filter size,
and it employs either the standard ReLU activation function (i=1) or the ReLU6 activation function (i=6). The decoding
route on the other side comprises of multiple decoding blocks (DB) for obtaining the segmentation mask.</figcaption>
</figure>

<figure>
  <p align="center">
  <img src="https://github.com/user-attachments/assets/95ac332f-e07f-4f09-b67d-0fd5420139fa" alt="Image Description" width="600" height="300">
</p>
  <figcaption>The architectural design of Dynamical Mobile Inverted Bottleneck Convolution (DMBC) which implements
Gating Signal along with depth wise separable convolution and squeeze and excite attention block. DMBC takes an input
tensor of shape (𝐻𝑖 × 𝑊𝑖 × 𝐶𝑖) and outputs a tensor of shape (𝐻𝑖 × 𝑊𝑖 × 𝐶𝑘).</figcaption>
</figure>
![dmbc]()

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
  <img src="https://github.com/user-attachments/assets/9e6fee04-657c-4403-a5b5-e79ffa4e5478" alt="Image Description">
</p>

## Citation
If you found REUnet useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

  ```bibtex
will be provided once published
```
 
