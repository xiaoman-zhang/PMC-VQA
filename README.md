# PMC-VQA
The official codes for [**PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering**](https://arxiv.org/pdf/2305.10415.pdf)  

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pmc-vqa-visual-instruction-tuning-for-medical/medical-visual-question-answering-on-pmc-vqa)](https://paperswithcode.com/sota/medical-visual-question-answering-on-pmc-vqa?p=pmc-vqa-visual-instruction-tuning-for-medical)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pmc-vqa-visual-instruction-tuning-for-medical/medical-visual-question-answering-on-vqa-rad)](https://paperswithcode.com/sota/medical-visual-question-answering-on-vqa-rad?p=pmc-vqa-visual-instruction-tuning-for-medical)


We propose a generative-based model for medical visual understanding by aligning visual information from a pre-trained vision encoder with a large language model, and establish a scalable pipeline to construct a large-scale medical visual question-answering dataset, named PMC-VQA, which contains 227k VQA pairs of 149k images that cover various modalities or diseases.

The dataset is available at [Huggingface](https://huggingface.co/datasets/xmcmic/PMC-VQA/)

The model checkpoints are available at [MedVInT-TE](https://huggingface.co/xmcmic/MedVInT-TE/) and [MedVInT-TD](https://huggingface.co/xmcmic/MedVInT-TD/).
**The previous checkpoint of MedVInT-TD was mistakenly uploaded. 
We have rectified the issue and updated the model's checkpoint on July 31. 
Now, you can access the correct and improved version of the model.**


- [PMC-VQA](#pmc-vqa)
  - [Usage](#usage)
    - [1. Create Environment](#1-create-environment)
    - [2. Prepare Dataset](#2-prepare-dataset)
    - [3. Model Checkpoints](#3-checkpoints)
  - [Acknowledgement](#acknowledgement)
  - [Contribution](#contribution)
  - [Cite](#cite)


## Usage

<!-- Repo Structure
```bash
PMC-VQA/: dataset
LLAMA_Model/: LLaMA pre-trained model path
src/:
    |--MedVInT_TD/: The code of the model of MedVInT-TD-MLP and MedVInT-TD-Transformer with LLaMA/PMC-LLaMA and CLIP/PMC-CLIP
    |   |--models/
    |   |   |--blocks.py 
    |   |   |--QA_model_mlp.py: model MedVInT-TD-MLP
    |   |   |--QA_model.py:  model MedVInT-TD-Transformer
    |   |   |--transformer.py 
    |   |--Results/: The checkpoints of the MedVInT_TD with PMC-LLaMA, PMC-CLIP on both blank and choice tasks
    |   |   |--blank_training/
    |   |   |--choice_training/
    |--MedVInT_TE/: The code of MedVInT-TE-Transformer with LLaMA/PMC-LLaMA and CLIP/PMC-CLIP
    |   |--dataset/
    |   |   |--dataset.py: Create dataset
    |   |   |--randaugment.py: data augmentation
    |   |--ds_config/
    |   |   |--ds_config_zero2.json: deep speed
    |   |--models/
    |   |   |--llama/
    |   |   |   |--blocks.py 
    |   |   |   |--vqa_model.py 
    |   |   |--pmcclip/: put the checkpoint of pmc_clip here
    |   |   |--pmc_oa/
    |   |   |   |--blocks.py 
    |   |   |   |--pmc_clip.py
    |   |   |   |--timm_model.py
    |   |   |   |--utils.py
    |   |--Results/: put the pre-trained checkpoint here
    |   |--README.md: README for MedVInT_TE
    |   |--train.py
    |   |--test.py
    |   |--train.sh
``` -->


### 1. Create Environment 

Please refer to https://github.com/chaoyi-wu/PMC-LLaMA

### 2. Prepare Dataset 

Download from [Huggingface](https://huggingface.co/datasets/xmcmic/PMC-VQA/) and save into ./PMC-VQA

### 3. Model Checkpoints

Download the pre-trained [MedVInT-TE](https://huggingface.co/xmcmic/MedVInT-TE/), and save into `./src/MedVInT_TE/Results `directly.  

Download the pre-trained [MedVInT-TD](https://huggingface.co/xmcmic/MedVInT-TD/), and save into `./src/MedVInT_TD/Results `directly.  

See [MedVInT_TE](./src/MedVInT_TE/README.md) and [MedVInT_TD](./src/MedVInT_TD/README.md)  for the details of training **MedVInT_TE** and **MedVInT_TD**. 

## Acknowledgement

CLIP -- https://github.com/openai/CLIP

PMC-CLIP -- https://github.com/WeixiongLin/PMC-CLIP

PMC-LLaMA -- [https://github.com/zphang/minimal-llama](https://github.com/chaoyi-wu/PMC-LLaMA)

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

## Contribution

Please raise an issue if you need help, any contributions are welcomed.

## Citation

If you use this code or use our pre-trained weights for your research, please cite our [paper](https://arxiv.org/abs/2305.10415)

```
@article{zhang2023pmcvqa,
      title={PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering}, 
      author={Xiaoman Zhang and Chaoyi Wu and Ziheng Zhao and Weixiong Lin and Ya Zhang and Yanfeng Wang and Weidi Xie},
      year={2023},
      journal={arXiv preprint arXiv:2305.10415},
}
```

