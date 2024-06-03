# NASA-F: FPGA-Oriented Search and Acceleration for Multiplication-Reduced Hybrid Networks.
This repository contains our PyTorch training code, evaluation code and pretrained models for NASA-F.

Our implementation is largely based on [AlphaNet](https://arxiv.org/pdf/2102.07954). 

For more details, please see [NASA-F: FPGA-Oriented Search and Acceleration for Multiplication-Reduced Hybrid Networks](https://www.semanticscholar.org/paper/NASA-F%3A-FPGA-Oriented-Search-and-Acceleration-for-Shi-Xu/5477af161258c9ab96cc6e495d88c61a507da7cd) by Huihong Shi, Yang Xu, Yuefei Wang, Wendong Mao, and Zhongfeng Wang.

If you find this repo useful in your research, please consider citing our work:

```BibTex
@article{shi2023nasa,
  title={NASA-F: FPGA-Oriented Search and Acceleration for Multiplication-Reduced Hybrid Networks},
  author={Shi, Huihong and Xu, Yang and Wang, Yuefei and Mao, Wendong and Wang, Zhongfeng},
  journal={IEEE Transactions on Circuits and Systems I: Regular Papers},
  year={2023},
  publisher={IEEE}
}
```

## Training
```python
python train.py --config-file configs/train_alphanet_models_add.yml
```

## Evolutionary Search
```python
python search.py --config-file configs/search_adder.yml
```



