# Scene Text Image Generation

This software implements the scene text image generation network based on GAN. For details, please refer to [our paper](https://link.springer.com/article/10.1007/s11063-019-10166-x).

## Scene Text Image Generation
* [Python 3.x](https://www.python.org/)
* [PyTorch 1.x](https://pytorch.org/)
* [TorchVision](https://pypi.org/project/torchvision/)
* [Pillow](https://pypi.org/project/Pillow/) 

## Data Preparation
Please convert your own training dataset according to the proposed one in samples/. There should be three sub-folders: image/ that contains real scene text images, mask/ that contains corresponding masks and masktr/ that contains masks in which the text area are dilated.

For testing, simply put the masks of scene text images in a folder.

## Train and Test
To train a new model, simply execute python train.py --data_path {train_data_path} --cuda. If you need to set other parameters, explore train.py for details.

To test a trained model, you need to explore and execute generate.py.

## Citation
    @article{gong2020generating,
      title={Generating Text Sequence Images for Recognition},
      author={Gong, Yanxiang and Deng, Linjie and Ma, Zheng and Xie, Mei},
      journal={Neural Processing Letters},
      pages={1--12},
      year={2020},
      publisher={Springer}
    }
