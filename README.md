# Image Translation models for Facades Generation :european_post_office: 


Hi! :wave: This is PyTorch implementation of CV approaches for image-to-image translation in the [Facades dataset](https://www.kaggle.com/datasets/balraj98/facades-dataset). Supported models are based on *pix2pix* framework ([Isola et al., 2017](https://arxiv.org/abs/1611.07004)):

- Deformable convolutions ([Zhu et al., 2018](https://arxiv.org/abs/1811.11168)) as the U-Net ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) backbone.
- Attention-based skip connection in the U-Net generator ([Oktay et al., 2018](https://arxiv.org/abs/1804.03999)).
- Pretrained segmentation models as generator module.

## Installation 

We suggest using the [requirements.txt](requirements.txt) to check and prepare all dependencies needed to run our code.

```shell 
pip3 install -r requirements.txt
```



## Training 

Our translation system requires a [segmenter model](models/segmenter.py) to assess about the generation correctness of facade images. The script [segment.py](segment.py) builds, trains and stores a segmentation model based on the FCN ([Long et al., 2014](https://arxiv.org/abs/1411.4038)) with VGGNet-16 pretrained weights ([Simonyan et al., 2014](https://arxiv.org/abs/1409.1556)).

```shell
python3 segment.py -n 5 -p results/segmenter/ -d cuda:0 --batch-size 5
```

At the end of the execution the segmenter files will be stored at [results/segmenter/](). The only required files to train the image-to-image translation models are the [centers.pt]() and [model.py]() files.

The [translate.py](translate.py) file runs the image translation system. Check the [report](report.pdf) to see a more detailed description about each model. The available choices are `base`, `deform`,  `attn`, `link`, `psp` and `fpn`:

```shell
python3 translate.py base -p results/base/ -s results/segmenter/ --data facades -d cuda:0
```

The results and image predictions will be stored at [results/base/]().



## Team :construction_worker:
- [Alejandro Dopico](https://github.com/AlejandroDopico2) ([alejandro.dopico2@udc.es](mailto:alejandro.dopico2@udc.es)).
- [Ana Ezquerro](https://anaezquerro.github.io) ([ana.ezquerro@udc.es](mailto:ana.ezquerro@udc.es)).

