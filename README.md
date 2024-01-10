# NCHO (ICCV 2023)

## [Project Page](https://taeksuu.github.io/ncho/) &nbsp;|&nbsp; [Paper](https://arxiv.org/pdf/2305.14345.pdf) 

![teaser.png](./assets/teaser1.png)

This is the official code for the ICCV 2023 paper "NCHO: Unsupervised Learning for Neural 3D Composition of Humans and Objects", a novel framework for learning a compositional generative model of humans and objects (backpacks, coats, scarves, and more) from real-world 3D scans.

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
python setup.py install
```

Download our pretrained models.
```
sh ./download_data.sh
```

Run one of the following commands and check the results in ./outputs/model

``Disentangled Control``
```
python test.py expname=200_backpack datamodule=ts_200_bag eval_mode=dis_hum # Same human + Different objects
python test.py expname=200_backpack datamodule=ts_200_bag eval_mode=dis_obj # Same object + Different humans
```

``Interpolation``
```
python test.py expname=200_backpack datamodule=ts_200_bag eval_mode=interp_hum # Same human + Interpolate objects
python test.py expname=200_backpack datamodule=ts_200_bag eval_mode=interp_obj # Same object + Interpolate humans
```

``Random Sampling``
```
python test.py expname=200_backpack datamodule=ts_200_bag eval_mode=sample # Random human + objects
```

## Dataset
We provide our captured raw 3D scans and the corresponding SMPL parameters. Folder "200" contains scans of the single person without any object. Other folders contain scans of the same person with many different objects.


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