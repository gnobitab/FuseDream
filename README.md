# FuseDream

This repo contains code for our paper:

**FuseDream: Training-Free Text-to-Image Generationwith Improved CLIP+GAN Space Optimization**

by *Xingchao Liu, Chengyue Gong, Lemeng Wu, Shujian Zhang, Hao Su and Qiang Liu*

![FuseDream](imgs/header_image.png?raw=true "FuseDream")

## Requirements
Please use `pip` or `conda` to install the following packages:
`PyTorch==1.7.1, torchvision==0.8.2, lpips==0.1.4` and also the requirements from [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch).

## Getting Started
Run the following command to generate images from text query:
`python fusedream_generator.py --text 'YOUR TEXT' --seed YOUR_SEED`

For example, to get an image of a blue dog:
`python fusedream_generator.py --text 'A photo of a blue dog.' --seed 1234`

The generated image will be stored in `./samples`
