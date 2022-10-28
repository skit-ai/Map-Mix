# Map-Mix

The official implementation of the method discussed in the paper **Improving Spoken Language Identification with Map-Mix**. 



<p align="center"><img src="images/model_final.png" width="35%"/></p>

**Abstract:**
The pre-trained multi-lingual XLSR model generalizes well for language identification after fine-tuning on unseen languages. However, the performance significantly degrades when the languages are not very distinct from each other, for example, in the case of dialects. Low resource dialect classification remains a challenging problem to solve. We present a new data augmentation method that leverages model training dynamics of individual data points to improve sampling for latent mixup. The method works well in low-resource settings where generalization is paramount. Our datamaps-based mixup technique, which we call Map-Mix improves weighted F1 scores by 2% compared to the random mixup baseline and results in a significantly well-calibrated model. 

<p align="center"><img src="images/datamaps.png" width="45%" /></p>

## Table of Content
  - [Requirements and Installation](#requirements-and-installation)
  - [Training and Evaluation](#training-and-evaluation)
  - [Results](#results)
  - [Licence](#licence)
  - [Citation](#citation)
  - [References](#references)


## Introduction
<p float="left">
  
   
</p>

<!--   -->

## Requirements and Installation
Each baseline methods mentioned in the paper(ie: pretrained-baselines, mixup-baselines, datamap-mixup, map-mix) are added to different branches of this repository.

- [Pretrained Baselines(wav2vec2, Hubert, XLSR)](https://github.com/skit-ai/Map-Mix/tree/baselines)
- [Mixup Baselines(static, latent-random, latent-within, latent-across)]()
- [Datamap Mixup (easy mixup, hard mixup, amb+easy mixup)]()
- [Map-Mix]()

Download the [LRE-2017 Dataset](https://www.nist.gov/system/files/documents/2017/09/29/lre17_eval_plan-2017-09-29_v1.pdf) and update the dataset path in config.json file.


```
git clone https://github.com/skit-ai/Map-Mix.git
cd Map-Mix

# checkout to the required method's branch
git checkout baselines
pip install -r requirements.txt
```
  
## Training and Evaluation
The training and testing script for all the baselines and methods are same. Edit the config.json file for the respective branch with the training hyperparameters and dataset path.

```
# Train the model
python train.py
```

```
# Evaluate the model and generate test prediction csv files
python test.py
```

```
# Get the evaluation metrics
# add the csv path to the test_metrics.py file
python test_metrics.py
```

## Results
<!-- ![](images/result1.png) -->
<p align="center"><img src="images/result1.png" width="45%" /></p>
<br><br>
<!-- ![](images/result2.png) -->
<p align="center"><img src="images/result2.png" width="45%" /></p>

## Licence
Nil
## Citation
Nil

## References
1. [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
2. [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://arxiv.org/abs/2009.10795)
3. [Exploiting Class Similarity for Machine Learning with Confidence Labels and Projective Loss Functions](https://arxiv.org/abs/2103.13607)
4. [S3PRL Github Repository](https://github.com/s3prl/s3prl)
5. [Huggingface's transformers package](https://github.com/huggingface/transformers)


