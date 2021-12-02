# FuseDream

This repo contains code for our paper:

**FuseDream: Training-Free Text-to-Image Generationwith Improved CLIP+GAN Space Optimization**

by *Xingchao Liu, Chengyue Gong, Lemeng Wu, Shujian Zhang, Hao Su and Qiang Liu*

![FuseDream](./imgs/header_img.png?raw=true "FuseDream")

## Requirements
Please use `pip` or `conda` to install the following packages:
`PyTorch==1.7.1, torchvision==0.8.2, lpips==0.1.4` and also the requirements from [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch).

## Getting Started

We transformed the pre-trained weights of BigGAN from TFHub to PyTorch. To save your time, you can download the transformed BigGAN checkpoints from:

https://drive.google.com/drive/folders/1nJ3HmgYgeA9NZr-oU-enqbYeO7zBaANs?usp=sharing

Put the checkpoints into `./BigGAN_utils/weights/`

Run the following command to generate images from text query:

`python fusedream_generator.py --text 'YOUR TEXT' --seed YOUR_SEED`

For example, to get an image of a blue dog:

`python fusedream_generator.py --text 'A photo of a blue dog.' --seed 1234`

The generated image will be stored in `./samples`

## Citations
If you use the code, please cite:

```text
@inproceedings{
brock2018large,
title={Large Scale {GAN} Training for High Fidelity Natural Image Synthesis},
author={Andrew Brock and Jeff Donahue and Karen Simonyan},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=B1xsqj09Fm},
}
```

and
```
```
