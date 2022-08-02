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

# Predict for new interactions

X_drug_pre, X_target_pre, y_pre = dataset.read_file_training_dataset_drug_target_pairs('./data/predict.txt')

df_data1, _, _, df_data2, _, _ = data_process(X_drug_pre, X_target_pre, y_pre,
                                 drug_encoding_1 = 'DGL_GCN', target_encoding_1 = 'CNN', 
                                 drug_encoding_2 = 'DGL_AttentiveFP', target_encoding_2 = 'Transformer',
                                 split_method = 'repurposing_VS')

model = models.model_pretrained('./model/')
y_pred, outputs = model.predict(df_data1, df_data2)
print(y_pred)
print(outputs)
