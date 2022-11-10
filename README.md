# Out-of-distribution Detection with Implicit Outlier Transformation



**[Out-of-distribution Detection with Implicit Outlier Transformation](https://openreview.net/forum?id=hdghx6wbGuD)**

ICLR 2023 (under review)

**Keywords**: Out-of-distribution Detection, Reliable Machine Learning

```
@inproceedings{
anonymous2023outofdistribution,
title={Out-of-distribution Detection with Implicit Outlier Transformation},
author={Anonymous},
booktitle={Submitted to The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=hdghx6wbGuD},
note={under review}
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



- [tiny-ImageNet-200](https://github.com/chihhuiho/CLAE/blob/main/datasets/download_tinyImagenet.sh)

- []
