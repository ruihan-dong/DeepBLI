# Two channels: GCN CNN with co-Attention, AFP Trans with co-Attention
# merge using element-wise product

import matplotlib
matplotlib.use("Agg")

# choose GPU number
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import DeepPurpose.dataset as dataset
import DTI_multi as models
from utils_multi import *

# Pre-train
X_drug, X_target, y = dataset.read_file_training_dataset_drug_target_pairs('./data/kinase.txt')

drug_encoding_1 = 'DGL_GCN'
target_encoding_1 = 'CNN'
drug_encoding_2 = 'DGL_AttentiveFP'
target_encoding_2 = 'Transformer'

train_1, val_1, test_1, train_2, val_2, test_2 = data_process(X_drug, X_target, y, 
                                drug_encoding_1, target_encoding_1, drug_encoding_2, target_encoding_2,
                                split_method='random',frac=[0.7,0.1,0.2])

config = generate_config(drug_encoding_1 = drug_encoding_1, 
                         target_encoding_1 = target_encoding_1, 
                         drug_encoding_2 = drug_encoding_2, 
                         target_encoding_2 = target_encoding_2, 
                         cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 25, 
                         test_every_X_epoch = 10, 
                         LR = 0.001, 
                         batch_size = 128,
                         hidden_dim_drug = 128,
                         transformer_emb_size_target = 128,
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12]                       
                        )
model = models.model_initialize(**config)
model.train(train_1, val_1, test_1, train_2, val_2, test_2)
model.save_model('./model/')


# Fine-tune and test
X_drug_ft, X_target_ft, y_ft = dataset.read_file_training_dataset_drug_target_pairs('./data/beta-lactamase.txt')
drug_encoding_1 = 'DGL_GCN'
target_encoding_1 = 'CNN'
drug_encoding_2 = 'DGL_AttentiveFP'
target_encoding_2 = 'Transformer'

train_1, val_1, test_1, train_2, val_2, test_2 = data_process(X_drug_ft, X_target_ft, y_ft, 
                                drug_encoding_1, target_encoding_1, drug_encoding_2, target_encoding_2,
                                split_method='random',frac=[0.7,0.1,0.2])

model = models.model_pretrained('./model/')
model.train(train_1, val_1, test_1, train_2, val_2, test_2)
model.save_model('./model/new')
