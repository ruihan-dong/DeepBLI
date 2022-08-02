# DeepBLI

**A transferable multi-channel model for detecting β-lactamase - inhibitor interaction**

#### Dependencies

Python3, PyTorch, RDkit, [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose)

#### Datasets

The label reversal KIBA dataset can be downloaded from [TransformerCPI](https://github.com/lifanchen-simm/transformerCPI).

Beta-lactamase dataset we collected can be found in this repository(./data/beta-lactamase.txt).

#### To use

##### Training

```bash
python3 train.py
```

DeepBLI is developed based on DeepPurpose library. Modification for multi-channel model are in `DTI_multi.py` and `utils_multi.py` files. `train.py` involves the pre-training and fine-tuning parts of code. 

##### Predicting

We also provide the trained  DeepBLI model in ./model. You can use the model to predict beta-lactamase - inhibitor interactions of interest.

1) Prepare the compound SMILES and protein sequence in ./data/predict.txt

Input format is like:

> CC(=O)c1c(O)c(C)c(O)c(Cc2c(O)c3C=CC(C)(C)Oc3c(C(=O)C=Cc3ccccc3)c2O)c1O MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW 1

The label of 0/1 is not associated with the predicting result.

2) Run the script

```bash
python3 predict.py
```

