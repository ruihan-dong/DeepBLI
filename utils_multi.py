import numpy as np
import pandas as pd
import torch
from torch.utils import data
import torch.nn.functional as F
import os

from DeepPurpose.utils import trans_protein, protein2emb_encoder, length_func, drug_2_embed, protein_2_embed

# random_fold
def create_fold(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac = test_frac, replace = False, random_state = fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]
    
    return train, val, test

def encode_drug(df_data, column_name = 'SMILES', save_column_name = 'drug_encoding'):
	print('encoding drug...')
	print('unique drugs: ' + str(len(df_data[column_name].unique())))
	df_data[save_column_name] = df_data[column_name]
	return df_data

def encode_protein(df_data, target_encoding, column_name = 'Target Sequence', save_column_name = 'target_encoding'):
	print('encoding protein...')
	print('unique target sequence: ' + str(len(df_data[column_name].unique())))
	if target_encoding == 'CNN':
		AA = pd.Series(df_data[column_name].unique()).apply(trans_protein)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
		# the embedding is large and not scalable but quick, so we move to encode in dataloader batch. 

	elif target_encoding == 'Transformer':
		AA = pd.Series(df_data[column_name].unique()).apply(protein2emb_encoder)
		AA_dict = dict(zip(df_data[column_name].unique(), AA))
		df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]

	return df_data

def data_process(X_drug = None, X_target = None, y=None, drug_encoding_1=None, target_encoding_1=None, 
				 drug_encoding_2=None, target_encoding_2=None, 
				 split_method = 'random', frac = [0.7, 0.1, 0.2], random_seed = 1, sample_frac = 1, mode = 'DTI', X_drug_ = None, X_target_ = None):
	
	if random_seed == 'TDC':
		random_seed = 1234
	DTI_flag = True

	print('Drug Target Interaction Prediction Mode...')
	if isinstance(X_target, str):
		X_target = [X_target]
	if len(X_target) == 1:
		# one target high throughput screening setting
		X_target = np.tile(X_target, (length_func(X_drug), ))

	df_data_1 = pd.DataFrame(zip(X_drug, X_target, y))
	df_data_2 = pd.DataFrame(zip(X_drug, X_target, y))
	df_data_1.rename(columns={0:'SMILES', 1: 'Target Sequence', 2: 'Label'}, inplace=True)
	df_data_2.rename(columns={0:'SMILES', 1: 'Target Sequence', 2: 'Label'}, inplace=True)
	print('in total: ' + str(len(df_data_1)) + ' drug-target pairs')
	'''
	if sample_frac != 1:
		df_data = df_data.sample(frac = sample_frac).reset_index(drop = True)
		print('after subsample: ' + str(len(df_data)) + ' data points...') 
	'''
	df_data_1 = encode_drug(df_data_1)
	df_data_1 = encode_protein(df_data_1, target_encoding_1)
	df_data_2 = encode_drug(df_data_2)
	df_data_2 = encode_protein(df_data_2, target_encoding_2)

	# dti split
	if DTI_flag:
		if split_method == 'repurposing_VS':
			pass
		else:
			print('splitting dataset...')

		if split_method == 'random': 
			train1, val1, test1 = create_fold(df_data_1, random_seed, frac)
			train2, val2, test2 = create_fold(df_data_2, random_seed, frac)

		elif split_method == 'repurposing_VS':
			train1 = df_data_1
			val1 = df_data_1
			test1 = df_data_1
			train2 = df_data_2
			val2 = df_data_2
			test2 = df_data_2	
	print('Done.')
	return train1.reset_index(drop=True), val1.reset_index(drop=True), test1.reset_index(drop=True), train2.reset_index(drop=True), val2.reset_index(drop=True), test2.reset_index(drop=True)

class data_process_loader(data.Dataset):

	def __init__(self, list_IDs_1, labels, df_1, list_IDs_2, df_2, **config):
		'Initialization'
		self.labels = labels
		self.list_IDs_1 = list_IDs_1
		self.df_1 = df_1
		self.list_IDs_2 = list_IDs_2
		self.df_2 = df_2
		self.config = config

		if self.config['drug_encoding_1'] == 'DGL_GCN':
			from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
			self.node_featurizer1 = CanonicalAtomFeaturizer()
			self.edge_featurizer1 = CanonicalBondFeaturizer(self_loop = True)
			from functools import partial
			self.fc1 = partial(smiles_to_bigraph, add_self_loop=True)

		if self.config['drug_encoding_2'] == 'DGL_AttentiveFP':
			from dgllife.utils import smiles_to_bigraph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
			self.node_featurizer2 = AttentiveFPAtomFeaturizer()
			self.edge_featurizer2 = AttentiveFPBondFeaturizer(self_loop=True)
			from functools import partial
			self.fc2 = partial(smiles_to_bigraph, add_self_loop=True)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs_1)

	def __getitem__(self, index):
		'Generates one sample of data'
		index = self.list_IDs_1[index]
		v_d1 = self.df_1.iloc[index]['drug_encoding']    
		v_d2 = self.df_2.iloc[index]['drug_encoding']
		v_d1 = self.fc1(smiles = v_d1, node_featurizer = self.node_featurizer1, edge_featurizer = self.edge_featurizer1)
		v_d2 = self.fc2(smiles = v_d2, node_featurizer = self.node_featurizer2, edge_featurizer = self.edge_featurizer2)
		v_p1 = self.df_1.iloc[index]['target_encoding']
		v_p2 = self.df_2.iloc[index]['target_encoding']
		v_p1 = protein_2_embed(v_p1)
		y = self.labels[index]
		return v_d1, v_p1, v_d2, v_p2, y

def generate_config(drug_encoding_1 = None, target_encoding_1 = None,
					drug_encoding_2 = None, target_encoding_2 = None, 
					result_folder = "./result/",
					input_dim_drug = 1024, 
					input_dim_protein = 8420,
					hidden_dim_drug = 256, 
					hidden_dim_protein = 256,
					cls_hidden_dims = [1024, 1024, 512],
					mlp_hidden_dims_drug = [1024, 256, 64],
					mlp_hidden_dims_target = [1024, 256, 64],
					batch_size = 256,
					train_epoch = 10,
					test_every_X_epoch = 20,
					LR = 1e-4,
					decay = 0,
					transformer_emb_size_drug = 128,
					transformer_intermediate_size_drug = 512,
					transformer_num_attention_heads_drug = 8,
					transformer_n_layer_drug = 8,
					transformer_emb_size_target = 64,
					transformer_intermediate_size_target = 256,
					transformer_num_attention_heads_target = 4,
					transformer_n_layer_target = 2,
					transformer_dropout_rate = 0.1,
					transformer_attention_probs_dropout = 0.1,
					transformer_hidden_dropout_rate = 0.1,
					mpnn_hidden_size = 50,
					mpnn_depth = 3,
					cnn_drug_filters = [32,64,96],
					cnn_drug_kernels = [4,6,8],
					cnn_target_filters = [32,64,96],
					cnn_target_kernels = [4,8,12],
					rnn_Use_GRU_LSTM_drug = 'GRU',
					rnn_drug_hid_dim = 64,
					rnn_drug_n_layers = 2,
					rnn_drug_bidirectional = True,
					rnn_Use_GRU_LSTM_target = 'GRU',
					rnn_target_hid_dim = 64,
					rnn_target_n_layers = 2,
					rnn_target_bidirectional = True,
					num_workers = 0,
					cuda_id = None,
					gnn_hid_dim_drug = 64,
					gnn_num_layers = 3,
					gnn_activation = F.relu,
					neuralfp_max_degree = 10,
					neuralfp_predictor_hid_dim = 128,
					neuralfp_predictor_activation = torch.tanh,
					attentivefp_num_timesteps = 2
					):

	base_config = {'input_dim_drug': input_dim_drug,
					'input_dim_protein': input_dim_protein,
					'hidden_dim_drug': hidden_dim_drug, # hidden dim of drug
					'hidden_dim_protein': hidden_dim_protein, # hidden dim of protein
					'cls_hidden_dims' : cls_hidden_dims, # decoder classifier dim 1
					'batch_size': batch_size,
					'train_epoch': train_epoch,
					'test_every_X_epoch': test_every_X_epoch, 
					'LR': LR,
					'drug_encoding_1': drug_encoding_1,
					'target_encoding_1': target_encoding_1, 
					'drug_encoding_2': drug_encoding_2,
					'target_encoding_2': target_encoding_2, 
					'result_folder': result_folder,
					'binary': False,
					'num_workers': num_workers,
					'cuda_id': cuda_id                 
	}
	if not os.path.exists(base_config['result_folder']):
		os.makedirs(base_config['result_folder'])
	

	if drug_encoding_1 == 'DGL_GCN':
		base_config['gnn_hid_dim_drug'] = gnn_hid_dim_drug
		base_config['gnn_num_layers'] = gnn_num_layers
		base_config['gnn_activation'] = gnn_activation
	if drug_encoding_2 == 'DGL_AttentiveFP':
		base_config['gnn_hid_dim_drug'] = gnn_hid_dim_drug
		base_config['gnn_num_layers'] = gnn_num_layers
		base_config['attentivefp_num_timesteps'] = attentivefp_num_timesteps
	else:
		raise AttributeError("Please use the correct drug encoding available!")
		
	if target_encoding_1 == 'CNN':
		base_config['cnn_target_filters'] = cnn_target_filters
		base_config['cnn_target_kernels'] = cnn_target_kernels
	if target_encoding_2 == 'Transformer':
		base_config['input_dim_protein'] = 4114
		base_config['transformer_emb_size_target'] = transformer_emb_size_target
		base_config['transformer_num_attention_heads_target'] = transformer_num_attention_heads_target
		base_config['transformer_intermediate_size_target'] = transformer_intermediate_size_target
		base_config['transformer_n_layer_target'] = transformer_n_layer_target	
		base_config['transformer_dropout_rate'] = transformer_dropout_rate
		base_config['transformer_attention_probs_dropout'] = transformer_attention_probs_dropout
		base_config['transformer_hidden_dropout_rate'] = transformer_hidden_dropout_rate
		base_config['hidden_dim_protein'] = transformer_emb_size_target
	else:
		raise AttributeError("Please use the correct protein encoding available!")

	return base_config


