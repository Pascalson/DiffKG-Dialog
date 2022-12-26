# DiffKG-Dialog
This repository aims to provide a PyTorch codebase for developing methods upon knowledge graph grounded dialogue generation on both pairwise and universal knowledge graphs.
It also serves as an implementation and includes the processed data of the paper ["Towards Large-Scale Interpretable Knowledge Graph Reasoning for Dialogue Systems." by Yi-Lin Tuan, Sajjad Beygi, Maryam Fazel-Zarandi, Qiaozi Gao, Alessandra Cervone, William Yang Wang, in the Findings of ACL 2022](https://arxiv.org/pdf/2203.10610)

The repository has three directories:
* Our processed datasets: `data/`
* For training/testing on datasets with paired knowledge graphs, such as SMD: `DiffKG_paired/`
* For training/testing on datasets with a universal knowledge graph, such as OpendialKG: `DiffKG_shared/`

## Installation
* We suggest to create a conda environment with python==3.7. We use [anaconda](https://www.anaconda.com/products/distribution).
* Inside your conda environment, install pytorch according to your CUDA version [refer to link](https://pytorch.org/get-started/previous-versions/)
* clone this repository, you might also need to refer to [Git LFS](https://git-lfs.com/) in order to download the data.

After activating the conda environment, do
```
pip install -r requirements.txt
pip install --no-deps datasets
```


## Preprocessed Datasets
We conducted experiments on three datasets, [stanford multi-domain dialouges (SMD)](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/), SMD-Reasoning (a modification of SMD), and [OpenDialKG](https://github.com/facebookresearch/opendialkg).
We build our datasets by converting output format, cleaning up some annotations and spliting if no splits are given.
Our processed datasets are in `data/`.


## Run the Code: Train and Evaluate a Model

### For datasets that each dialogue has paired knowledge
* This directory is for the case where each dialogue is paired with different knowledge graph, e.g, the SMD dataset.
To start this type of experiment, please change to the directory first.
```
cd DiffKG_paired
```

* To train and evaluate the proposed model DiffKG on **SMD**, one can run:
```
python train.py --data_dir ../data/smd --output_dir exps/smd_diffkgT5
python evaluate.py --data_dir ../data/smd --model_dir exps/smd_diffkgT5+[timestamp]
# note that the timestamp is auto-generated when saving the trained models, so just refer to the saved name
```

* To train and evaluate the proposed model DiffKG on **SMD-Reasoning**, one can run:
```
python train.py --data_dir ../data/smd_reasoning --output_dir exps/smd-reasoning_diffkgT5 --only_reasoning_data
python evaluate.py --data_dir ../data/smd_reasoning --model_dir exps/smd-reasoning_diffkgT5+[timestamp] --only_reasoning_data
```


### For datasets that all dialogues share the same knowledge
* This directory is for the case where all dialogues are paired with the same knowledge graph, e.g, the OpenDialKG dataset.
To start this type of experiment, please change to the directory first.
```
cd DiffKG_shared
```

* To train and evaluate the proposed model DiffKG on **OpenDialKG**, one can run:
```
python train.py --data_dir ../data/opendialkg --output_dir exps/opendialkg_diffkgT5
python evaluate.py --data_dir ../data/opendialkg --model_dir exps/opendialkg_diffkgT5+[timestamp]
```

### Reading the Results
* All the evaluation results will be saved in the specified `--model_dir` in the evaluation script. The results files saved are:
  * `evaluation.json`
  * `visualization.txt`

## Other Notes
When running this code, some points may need to be taken care of, e.g., the inference method in the current code is naive greedy search, it does not automatically select checkpoint but uses the final epoch one unless manually change it, it does not use fixed random seed.

## Citaion
If you find this repository helpful, please kindly cite the paper.
```
@inproceedings{tuan2022towards,
  title={Towards Large-Scale Interpretable Knowledge Graph Reasoning for Dialogue Systems},
  author={Tuan, Yi-Lin and Beygi, Sajjad and Fazel-Zarandi, Maryam and Gao, Qiaozi and Cervone, Alessandra and Wang, William Yang},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2022},
  pages={383--395},
  year={2022}
}
```
