# Out-of-distribution Detection with Implicit Outlier Transformation

**[Out-of-distribution Detection with Implicit Outlier Transformation](https://openreview.net/forum?id=hdghx6wbGuD)**   (ICLR 2023)

Qizhou Wang, Junjie Ye, Feng Liu, Quanyue Dai, Marcus Kalander, Tongliang Liu, Jianye Hao, and Bo Han. 




**Keywords**: Out-of-distribution Detection, Reliable Machine Learning

```
@inproceedings{
wang2023outofdistribution,
title={Out-of-distribution Detection with Implicit Outlier Transformation},
author={Qizhou Wang and Junjie Ye and Feng Liu and Quanyu Dai and Marcus Kalander and Tongliang Liu and Jianye Hao and Bo Han},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=hdghx6wbGuD}
}
```

**Abstract**: Outlier exposure (OE) is powerful in out-of-distribution (OOD) detection, enhancing detection capability via model fine-tuning with surrogate OOD data. However, surrogate data typically deviate from test OOD data. Thus, the performance of OE when facing unseen OOD data, can be weaken. To address this issue, we propose a novel OE-based approach that makes the model perform well for unseen OOD situations, even for unseen OOD cases. It leads to a min-max learning scheme---searching to synthesize OOD data that leads to worst judgments and learning from such OOD data for the uniform performance in OOD detection. In our realization, these worst OOD data are synthesized by transforming original surrogate ones, where the associated transform functions are learned implicitly based on our novel insight that model perturbation leads to data transformation. Our methodology offers an efficient way of synthesizing OOD data, which can further benefit the detection model, besides the surrogate OOD data. We conduct extensive experiments under various OOD detection setups, demonstrating the effectiveness of our method against its advanced counterparts.

## Get Started

### Environment
- Python (3.7.10)
- Pytorch (1.7.1)
- torchvision (0.8.2)
- CUDA
- Numpy

### Pretrained Models and Datasets

Pretrained models are provided in folder

```
./ckpt/
```

Please download the datasets in folder

```
../data/
```

Surrogate OOD Dataset

- [tiny-ImageNet-200](https://github.com/chihhuiho/CLAE/blob/main/datasets/download_tinyImagenet.sh)


Test OOD Datasets 

- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

- [Places365](http://places2.csail.mit.edu/download.html)

- [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)
 
- [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

- [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)


### File Structure

After the preparation work, the whole project should have the following structure:

```
./DOE
├── README.md
├── ckpt                            # datasets
│   ├── cifar10_wrn_pretrained_epoch_99.pt 
│   └── cifar100_wrn_pretrained_epoch_99.pt
├── models                          # models
│   └── wrn.py
├── utils                           # utils
│   ├── display_results.py                        
│   ├── utils_awp.py
│   └── validation_dataset.py
└── doe_final.py                    # training code
```



## Training

To train the DOE model on CIFAR benckmarks, simply run:

- CIFAR-10
```train cifar10
python doe_final.py cifar10 
```


- CIFAR-100
```train cifar100
python doe_final.py cifar100
```

## Results

The key results on CIFAR benchmarks are listed in the following table. 

|     | CIFAR-10 | CIFAR-10 | CIFAR-100 | CIFAR-100 |
|:---:|:--------:|:--------:|:---------:|:---------:|
|     |   FPR95  |   AUROC  |   FPR95   |   AUROC   |
| MSP |   53.77  |   88.40  |   76.73   |   76.24   |
|  OE |   12.41  |   97.85  |   45.68   |   87.61   |
| DOE |   **5.15**   |   **98.78**  |   **25.38**   |   **93.97**   |
