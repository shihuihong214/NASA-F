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

## Evaluation
To reproduce our results:
- Please first download our [pretrained AlphaNet models](https://drive.google.com/file/d/1CyZoPyiCoGJ0qv8bqi7s7TQRUum_8FeG/view?usp=sharing) from a Google Drive path and put the pretrained models under your local folder *./alphanet_data*

- To evaluate our pre-trained AlphaNet models, from AlphaNet-A0 to A6, on ImageNet with a single GPU, please run:

    ```python
    python test_alphanet.py --config-file ./configs/eval_alphanet_models.yml --model a[0-6]
    ```

    Expected results:
    
    | Name  | MFLOPs  | Top-1 (%) |
    | :------------ |:---------------:| -----:|
    | AlphaNet-A0      | 203 | 77.87 |
    | AlphaNet-A1     | 279 | 78.94 |
    | AlphaNet-A2     | 317 | 79.20 |
    | AlphaNet-A3    | 357 | 79.41 |
    | AlphaNet-A4     | 444 | 80.01 |
    | AlphaNet-A5 (small)     | 491 | 80.29 |
    | AlphaNet-A5 (base)    | 596 | 80.62 |
    | AlphaNet-A6     | 709 | 80.78 |
    
- Additionally, [here](https://drive.google.com/file/d/1NgZhJy8MJnuxjXkJ0gfnBGyrUVYwbAmx/view?usp=sharing) is our pretrained supernet with KL based inplace-KD and [here](https://drive.google.com/file/d/1rj1opDnlBD2_8ZV--LUSn8HXWfhiMdu8/view?usp=sharing) is our pretrained supernet without inplace-KD. 

## Training
To train our AlphaNet models from scratch, please run:
```python
python train_alphanet.py --config-file configs/train_alphanet_models.yml --machine-rank ${machine_rank} --num-machines ${num_machines} --dist-url ${dist_url}
```
We adopt SGD training on 64 GPUs. The mini-batch size is 32 per GPU; all training hyper-parameters are specified in [train_alphanet_models.yml](configs/train_alphanet_models.yml).

## Evolutionary search
In case you want to search the set of models of your own interest - we provide an example to show how to search the Pareto models for the best FLOPs vs. accuracy tradeoffs in _parallel_supernet_evo_search.py_; to run this example:
```python
python parallel_supernet_evo_search.py --config-file configs/parallel_supernet_evo_search.yml 
```

## License
AlphaNet is licensed under CC-BY-NC.

## Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING](CONTRIBUTING.md) and [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) for more info.


