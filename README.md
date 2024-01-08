# NCHO (ICCV 2023)

## [Project Page](https://taeksuu.github.io/ncho/) &nbsp;|&nbsp; [Paper](https://arxiv.org/pdf/2305.14345.pdf) 

![teaser.png](./assets/teaser1.png)

This is the official code for the ICCV 2023 paper "NCHO: Unsupervised Learning for Neural 3D Composition of Humans and Objects", a novel framework for learning a compositional generative model of humans and objects (backpacks, coats, scarves, and more) from real-world 3D scans.

## Dataset
We provide our captured raw 3D scans and the corresponding SMPL parameters used in the paper. Folder "200" contains scans of the person without any object. Other folders contain scans of the same person with many different objects.

### Agreement
The NCHO dataset is available for non-commercial research purposes only. Any other use, in particular any use for commercial purposes, is prohibited.

## Quick Start

Clone the repository.
```
git clone https://github.com/taeksuu/ncho.git
cd ncho
```

Setup the environment using conda.
```
conda env create -f environment.yaml
conda activate ncho
```

Download our pretrained models.
```
sh ./download_data.sh
```


## Citation

If you use this code or dataset for your research, please cite our paper:

```
@InProceedings{Kim_2023_ICCV,
    author    = {Kim, Taeksoo and Saito, Shunsuke and Joo, Hanbyul},
    title     = {NCHO: Unsupervised Learning for Neural 3D Composition of Humans and Objects},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {14817-14828}
}
```

## Thanks to
- https://github.com/xuchen-ethz/gdna