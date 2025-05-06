# [NoWag: A Unified Framework for Shape Preserving Compression of Large Language Models](https://arxiv.org/abs/2504.14569)
[Lawrence Liu](https://lawrencerliu.github.io)<sup>1</sup>, [Inesh Chakrabarti](https://www.linkedin.com/in/inesh-chakrabarti-878602183/)<sup>1</sup>, [Yixiao Li](https://yxli2123.github.io/)<sup>2</sup>, [Mengdi Wang](https://mwang.princeton.edu/)<sup>3</sup>, [Tuo Zhao](https://www2.isye.gatech.edu/~tzhao80/)<sup>2</sup>, [Lin F. Yang](http://drlinyang.net/)<sup>1</sup><br>
<sup>1</sup> UCLA, <sup>2</sup> Georgia Tech, <sup>3</sup> Princeton University<br>

![Nowag Overview](assets/Overview.png)

## Overview
This is the official implementation of NoWag, a unified framework for shape-preserving compression of large language models. 

## Features
- **No**rmalized **W**eight and **A**ctivation **G**uided Compression (**NoWag**) is a family of computationally efficient pruning (**NoWag-P**) and quantization (**NoWag-VQ**) algorithms for LLMs with a shared normalization method and optimization objective. 
-  **NoWag-P** and **NoWag-VQ** perform competitively against SOTA pruning and compression algorithms.
- **NoWag-VQ** demonstrating reduced calibration data dependence compared with SOTA VQ Methods

## Requirements
- Python 3.13.2+
- Miniconda/Anaconda
- Cuda

## Installation
Clone the repository and install the required dependencies:
```bash
git clone git@github.com:LawrenceRLiu/NoWAG.git
cd NoWag
conda env create -f env.yml
conda activate NoWag
```

## Usage
### One Shot Compression
One shot compression is a method that allows for the compression of large language models in a single pass, without the need for iterative training or fine-tuning. This is the most computationally efficient method of compression.

We use the average l2 norm of the sample activations, also know as the diagonals of the hessians, to provide data awareness during one shot compression. This can be done by running `scripts/generate_hessians.py`, or alternatively, we have provided a bash script to generate the hessians for the Llama 2 7B/13B/70B and the Llama-3 8B/70B models using the same seed and 
calibration data as in the paper. The script can be found in `scripts/generate_hessians.bash`.

To perform one-shot compression, run the `NoWag.py` script. We use [Hydra](https://github.com/facebookresearch/hydra) for configuration management, so you can specify the model to compress, compression method, parameters, etc. By default the script will run NoWag-P on the Llama 2 7B model.
Currently NoWag supports two paradigms of shape preserving compression:

2. **Pruning (NoWag-P)**: This method prunes the model weights based on their importance. To run add `compress=prune` to the command line.  This is the default method used in the `NoWag.py` script. We support both unstructured and N:M pruning. The default is unstructured pruning. To run N:M pruning, add `+compress.kwargs.pattern=[$N:$M]` to the command line. We have provided a bash script to run NoWag-P on the Llama 2 7B/13B/70B and the Llama-3 8B/70B models using the same seed and calibration data as in the paper. The script can be found in `scripts/prune.bash`.

1. **Vector Quantization (NoWag-VQ)**: This method quantizes the model weights using vector quantization techniques. We have provided a bash script to run NoWag-VQ on the Llama 2 7B/13B/70B and the Llama-3 8B models using the same seed and calibration data as in the paper. The script can be found in `scripts/quantize.bash`. We are working on implementing the cuda kernel for inference, the current results are from simulation only, and cannot be used for inference.

### Layerwise Fine Tuning
We also examine the performance of \methodVQ beyond the ``one-shot'' compression regime. Existing literature has proposed several methods for post quantization finetuning. One popular method is finetuning the remaining continuous parameters of each transformer block to minimize the block output errors. We implemented this method in our codebase, to run this method, 
run `finetune_layerwise.py` with `run_name` specified to be the same as the `run_name` used in the `NoWag.py` script. 

## Repository Structure
```
.
├── models/             # Model definitions
├── config/            # Configuration files
├── scripts/            # Utility scripts
├── src            # src code
├── NoWag.py         # Main script for NoWag compression
├── finetune_layerwise.py # Script for layerwise fine tuning
└── README.md           # Project documentation
```

## ToDOs:
- [ ] Cuda Kernels for NoWag-VQ (Comming Soon!)
- [ ] End to End Post Compression Fine Tuning (Comming Soon!)
- [ ] [Trellis](https://arxiv.org/abs/2406.11235) Quantization (Comming Soon!)
- [ ] Support for more models (Comming Soon!)

## Citation
If you use this framework in your research, please cite:
```
@misc{liu2025nowagunifiedframeworkshape,
      title={NoWag: A Unified Framework for Shape Preserving Compression of Large Language Models}, 
      author={Lawrence Liu and Inesh Chakrabarti and Yixiao Li and Mengdi Wang and Tuo Zhao and Lin F. Yang},
      year={2025},
      eprint={2504.14569},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.14569}, 
}
```

## License

This project is licensed under the GNU GPL v3 License. See the [LICENSE](LICENSE) file for details. Use of Llama models is governed by the Meta license available [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

## Contact
For questions or issues, please contact [lawrencerliu@ucla.edu](mailto:lawrencerliu@ucla.edu).
