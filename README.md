# Improving voice conversion with Extremal Neural Optimal Transport

This is a fork of [kNN-VC](https://github.com/bshall/knn-vc), which uses Extremal Neural Optimal Transport instead of Nearest Neighbors during sample matching in Voice Conversion.

Links:

- kNN-VC paper: [https://arxiv.org/abs/2305.18975](https://arxiv.org/abs/2305.18975)
- XNOT paper: [https://arxiv.org/abs/2301.12874](https://arxiv.org/abs/2301.12874)


![kNN-VC overview](./pics/knn-vc.png)

Figure: kNN-VC setup. The source and reference utterance(s) are encoded into self-supervised features using WavLM. Each source feature is assigned to the mean of the k closest features from the reference. The resulting feature sequence is then vocoded with HiFi-GAN to arrive at the converted waveform output.

![XNOT method](./pics/OT_map_def_perfect_v6.png)
Figure: XNOT setup. By computing incomplete transport (IT) maps in high dimensions with neural networks, XNOT algorithm can partially align distributions or approximate extremal (ET) transport maps for unpaired domain translation tasks. 

## Quickstart

1. Clone [this](https://github.com/tardis-forever/xnot-vc) repo
2. **Install dependencies** from `requirements.txt`. It is advised that you have python version 3.10 or greater, and torch version v2.0 or greater.
3. Run reproducible experiments from [xnot_demo](./xnot_demo.ipynb) 


## Repository structure

Additions to original repo:
- ```xnot.py``` - implementation of XNOT module for general domain translation task
- ```xnot_matcher.py``` - modification of `KNeighborsVC` with `XNOT` mapping support
- ```xnot_demo.ipynb``` - replication notebook with experiments

## Datasets
- [LibriSpeech (test-clean)](http://www.openslr.org/12) should be placed in the root of repository;
- [LibriSpeech Alignments](https://github.com/CorentinJ/librispeech-alignments) should be placed in the root of repository.


## Experiments

### Basic setup
### Ablation setup
### V2 setup
### Cross-lingual translation

During inference - voice conversion - an instance of `XNot` is trained (and returned) as an approximation of the best transport map between source and target audio features distribution.
 // todo

## Performance

All experiments were run on single V-100 GPU.

For intelligibility metrics (`WER`, `CER`) average values over all generated samples are reported.
The performance on the LibriSpeech dev-clean set is summarized (all models use [prematched HiFiGAN](https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt)):

### Basic setup

| model       | w | WER (%) &darr; | CER (%) &darr; | EER (%) &uarr; |
| ----------- |:-:|:--------------:|:--------------:|:--------------:|
| kNN-VC*     | - | 6.29*          | 2.34*          | 35.73*         |
| kNN-VC      | - | **10.58**      | **3.53**       | 90.99          |
| XNOT-VC     | 1 | 11.32          | 3.97           | **92.22**      |
| XNOT-VC     | 2 | 11.32          | 3.97           | **92.67**      |
| XNOT-VC     | 4 | 11.32          | 3.97           | 90.22          |
|

*As reported by original authors on `dev-clean` split in original `README.md`, `EER` was calculated in a different unspecified manner and reportedly capped at 0.5.  
As in the 4.3. section of [original paper](https://arxiv.org/abs/2305.18975) authors mention `test-clean` split, we chose 
it as the evaluation set in our research.

### Ablation setup

| model       | w | WER (%) &darr; | CER (%) &darr; | EER (%) &uarr; |
| ----------- |:-:|:--------------:|:--------------:|:--------------:|
| XNOT-VC*    | 1 | 11.32          | 3.97           | **92.22**      |
| XNOT-VC*    | 2 | 11.32          | 3.97           | **92.67**      |
| XNOT-VC*    | 4 | 11.32          | 3.97           | 90.22          |
| XNOT-VC-v2* | 1 | 11.02          | **3.83**       | 91.44          |
| XNOT-VC-v2* | 2 | 11.02          | **3.83**       | 91.11          |
| XNOT-VC-v2* | 4 | 11.02          | **3.83**       | 90.44          |
| XNOT-VC-rec | 1 | 17.35          | 7.2            | 90.00          |
| XNOT-VC-rec | 2 | 17.35          | 7.2            | 89.25          |
| XNOT-VC-rec | 4 | 17.35          | 7.2            | 89.25          |


*Provided for comparison.

### V2 setup

| model       | w | WER (%) &darr; | CER (%) &darr; | EER (%) &uarr; |
| ----------- |:-:|:--------------:|:--------------:|:--------------:|
| kNN-VC*     | - | **10.58**      | **3.53**       | 90.99          |
| kNN-VC-v2   | - | **10.58**      | **3.53**       | 91.11          |
| XNOT-VC*    | 1 | 11.32          | 3.97           | 92.22          |
| XNOT-VC*    | 2 | 11.32          | 3.97           | 92.67          |
| XNOT-VC*    | 4 | 11.32          | 3.97           | 90.22          |
| XNOT-VC-v2  | 1 | 11.02          | 3.83           | **91.44**      |
| XNOT-VC-v2  | 2 | 11.02          | 3.83           | **91.11**      |
| XNOT-VC-v2  | 4 | 11.02          | 3.83           | 90.44          |

*Provided for comparison.

### Cross-lingual translation

| model       | w | WER (%) &darr; | CER (%) &darr; | EER (%) &uarr; |
| ----------- |:-:|:--------------:|:--------------:|:--------------:|
| kNN-VC-ru   | - | **15.37**      | **7.95**       | 92.67          |
| XNOT-VC-ru  | 1 | 15.84          | 8.38           | **94.67**      |
| XNOT-VC-ru  | 2 | 15.84          | 8.38           | **94.67**      |
| XNOT-VC-ru  | 4 | 15.84          | 8.38           | **94.00**      |


## Results

// todo

### Basic setup
### Ablation setup
### V2 setup
### Cross-lingual translation


## Generated audios

All audios generated during experiments are available on [YandexDisk](https://disk.yandex.ru/d/-qarNdQQkdMKEw).

## Credits
- [X-vectors for speaker verification](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) developer tools for machine learning;
- [kNN-VC](https://github.com/bshall/knn-vc) original paper;
- [XNOT](https://github.com/milenagazdieva/ExtremalNeuralOptimalTransport) original paper.
