# RPNSIC

This repository contains the code for reproducing the results with the training file, in the following paper:

_Exploring Resolution Fields for Scalable Image Compression with Uncertainty Guidance_

# 1. Preparation

## Dataset
1. Download [CLIC2020 Dateset](https://tensorflow.google.cn/datasets/catalog/clic) and place them in 'Dataset\'.
2. You can generate the patches using the patch generation file from [Link](https://github.com/liujiaheng/CompressionData).
3. Downsample the training images to size HxW, 2Hx2W and 4Hx4W.
4. Generate the tfrecord for spatial scalable.
```
python create_tfrecords.py --train_tfrecords ./xxx.tfrecords --input_image ./your_4Hx4W_image_folder, --input_image_half ./your_2Hx2W_image_folder, --input_image_quater ./your_4Hx4W_image_folder
```

## Environment

* Python==3.6.13

* Tensorflow==1.15.0

* [Tensorflow-Compression](https://github.com/tensorflow/compression)==1.3

# 2. Train

If your machine has multiple GPUs, you can select which GPU you want to run on by setting the environment variable before the Python operation
```
CUDA_VISIBLE_DEVICES=0 (0, 1, 2, 3,...)
```

Before training, modify the arguments like train_dataset, checkpoint_dir, num_filter, and lambda.

## Spatial Scalable
```
python train_spaital.py
```

## Quality Scalable
```
python train_quality.py
```

# 3. Test

* When running the `evaluate()`, the input image will be compressed into several bitstreams like: stream_B.tfci and stream_e1.tfci, it will also evaluate the bpp and PSNR (or MS-SSIM) of each layer.

* When running the `decompress()` the bitstreams can be decoded to images.

Before testing, modify the arguments like input_image, output_folder, checkpoint_dir, and num_filters.

## Spatial Scalable
```
python eval_spaital.py
```

## Quality Scalable
```
python eval_quality.py
```

## Contact
If you find any problem in the code and want to ask any questions, please send us an email
dyzhang@bjtu.edu.cn
