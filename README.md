# DeepBLI

**A transferable multi-channel model for detecting β-lactamase - inhibitor interaction**

#### Dependencies

Python3, PyTorch, RDkit, [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose)

#### Datasets

The label reversal KIBA dataset can be downloaded from [TransformerCPI](https://github.com/lifanchen-simm/transformerCPI).

Beta-lactamase dataset we collected can be found in this repository.

#### To use

DeepBLI is developed based on DeepPurpose library. Modification for multi-channel model are in `DTI_multi.py` and `utils_multi.py` files. So first please copy these two files to yourpath/DeepPurpose.

`run.py` involves the pre-training, fine-tuning and predicting parts of code. We also provide the trained  DeepBLI model.