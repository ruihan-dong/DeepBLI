# DeepBLI

**A transferable multi-channel model for detecting β-lactamase - inhibitor interaction**

## Dependencies

Python3, PyTorch (>=1.9.0), RDkit, [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose), DGL

## Datasets

The label reversal KIBA dataset can be downloaded from [TransformerCPI](https://github.com/lifanchen-simm/transformerCPI).

Beta-lactamase dataset we collected can be found in this repository(./data/beta-lactamase.txt).

## To use

### Training

```bash
python train.py
```

DeepBLI is developed based on DeepPurpose library. Modification for multi-channel model are in `DTI_multi.py` and `utils_multi.py` files. `train.py` involves the pre-training and fine-tuning parts of code. 

### Predicting

We also provide the trained  DeepBLI model in ./model. You can use the model to predict beta-lactamase - inhibitor interactions of interest.

1) Prepare the compound SMILES and protein sequence in ./data/predict.txt

Input format is like:

> CC(=O)c1c(O)c(C)c(O)c(Cc2c(O)c3C=CC(C)(C)Oc3c(C(=O)C=Cc3ccccc3)c2O)c1O MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW 1

The label of 0/1 is not associated with the predicting result.

2) Run the script

```bash
python predict.py
```
## To cite
```text
@article{doi:10.1021/acs.jcim.2c01008,
author = {Dong, Ruihan and Yang, Hongpeng and Ai, Chengwei and Duan, Guihua and Wang, Jianxin and Guo, Fei},
title = {DeepBLI: A Transferable Multichannel Model for Detecting β-Lactamase-Inhibitor Interaction},
journal = {Journal of Chemical Information and Modeling},
volume = {62},
number = {22},
pages = {5830-5840},
year = {2022},
doi = {10.1021/acs.jcim.2c01008},
note ={PMID: 36245217}
}
```
