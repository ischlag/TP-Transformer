# TP-Transformer
This repository contains the source code for the paper [*Enhancing the Transformer with Explicit Relational Encoding for Math Problem Solving*](https://arxiv.org/abs/1910.06611). The repository contains all the code necessary to reproduce the performance on the Deepmind Mathematics dataset (dm_math). We provide downloadlinks for the preprocessed dataset and several pretrained models. 

## Google Colab Notebook
We prepared a [Google Colab notebook](https://colab.research.google.com/drive/1Zi9FOwcO_i-4FDQQGgMMfcZxZISpJNMT) for anyone to experiment first hand with the TP-Transformer.

## Train from Scratch
## Requirements
```
pip3 install --upgrade gdown
pip3 install --upgrade torch==1.1.0
pip3 install --upgrade torchtext==0.3.1
pip3 install --upgrade tensorboardX==1.8
```

### Download the dataset and pretrained models.
Preprocessing the dataset takes a while so you maybe want to download the already preprocessed dataset. 
```
wget -O data.tar.gz https://zenodo.org/record/3532678/files/data.tar.gz?download=1
wget -O pretrained.tar.gz https://zenodo.org/record/3532678/files/pretrained.tar.gz?download=1
```

### Usage
```
python3 main.py --help
```
The script supports multi-gpu training, gradient accumulation, and two different data pipelines. We also provide the scripts that we used in order to preprocess and merge the dm_math modules into one big module called *all_modules*.

## Citation
```
@article{schlag2019enhancing,
  title={Enhancing the Transformer with Explicit Relational Encoding for Math Problem Solving},
  author={Schlag, Imanol and Smolensky, Paul and Fernandez, Roland and Jojic, Nebojsa and Schmidhuber, J{\"u}rgen and Gao, Jianfeng},
  journal={arXiv preprint arXiv:1910.06611},
  year={2019}
}
```
