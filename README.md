# CPA-Enhancer: Chain-of-Thought Prompted Adaptive Enhancer for Object Detection under Unknown Degradations

This is the official repository of the paper: [CPA-Enhancer: Chain-of-Thought Prompted Adaptive Enhancer for Object Detection under Unknown Degradations](https://arxiv.org/abs/2403.11220)  for **segmentation tasks**.



## 🛠️ Installation

- **Step0**. Download and install [Miniconda](https://docs.anaconda.com/free/miniconda/) from the official website.
- **Step1**. Create a conda environment and activate it.

```shell
conda create --name mmseg python=3.8 -y
conda activate mmseg
```

- **Step2**.Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

```shell
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

- **Step3**. Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

- **Step4**. Install other related packages

```shell
cd CPA_Seg
pip install -r ./cpa/requirements.txt
```

## 📁 Data Preparation

You can download our reorganized [ACDC](https://acdc.vision.ee.ethz.ch/) datase (in cityscapes format) from this [link](https://pan.baidu.com/s/1qeETFeJ_azR2t00VrOipSg). (pwd: `tuag`) 
Remember to modify the `data_root` in `configs/__base__/datasets/cityscapes.py`.

## 🎯 Usage

- Recompile the code.

```python
pip install -v -e .
```

- Training

```python
# CPA-Enhancer + deeplabv3plus
python tools/train.py configs/deeplabv3plus/acdc_deeplabv3plus_config.py
```

```python
# CPA-Enhancer + segformer
python tools/train.py  configs/segformer/acdc_segformer_config.py
```

- Testing

```python
# CPA-Enhancer + deeplabv3plus
python tools/test.py configs/deeplabv3plus/acdc_deeplabv3plus_config.py cpa/pretrained_models/deeplabv3plus.epoch
```

```python
# CPA-Enhancer + segformer
python tools/test.py  configs/segformer/acdc_segformer_config.py cpa/pretrained_models/segformer.epoch
```

- Inference

```python
python demo/image_demo.py \
	--img path/to/testimg.png  # path to your input image
	--config path/to/configfile # Eg. ..configs/segformer/acdc_segformer_config.py 
	--weights path/to/pretrained_models/xx.pth 
```

You can download our pretrained models from this [link](https://pan.baidu.com/s/1AZrH1fcdB7XAG9WwqNa-8Q). (pwd: `m1r1`)



## 📊 Results
We cascade our proposed CPA-Enhancer with two basic segmentation models, DeepLabv3+ and Segformer, and labeled them as Ours(D) and Ours(S) respectively. 
### Quantitative results
<p align="center">
  <img src="https://github.com/zyw-stu/CPA-Seg/blob/main/cpa/pics/test_seg.png" alt="Overall Workflow of the CPA-Enhancer Framework" style="width:80%">
  <br>
  <em>Quantitative comparisons on the ACDC test set.</em>
</p>

### Visual Results
<p align="center">
  <img src="https://github.com/zyw-stu/CPA-Seg/blob/main/cpa/pics/seg.jpg" alt="Overall Workflow of the CPA-Enhancer Framework" style="width:80%">
  <br>
  <em>Qualitative comparisons of semantic segmentation on the ACDC validation set. Zoom in on the colored annotation boxes to better observe the differences.</em>
</p>

## 💐 Acknowledgments

Special thanks to the creators of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) upon which this code is built, for their valuable work in advancing object detection research.

## 🔗 Citation

If you use this codebase, or CPA-Enhancer inspires your work, we would greatly appreciate it if you could star the repository and cite it using the following BibTeX entry.
```
@misc{zhang2024cpaenhancer,
      title={CPA-Enhancer: Chain-of-Thought Prompted Adaptive Enhancer for Object Detection under Unknown Degradations}, 
      author={Yuwei Zhang and Yan Wu and Yanming Liu and Xinyue Peng},
      year={2024},
      eprint={2403.11220},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
