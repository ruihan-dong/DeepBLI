import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from DeepPurpose.pybiomed_helper import _GetPseudoAAC, CalculateAADipeptideComposition, \
calcPubChemFingerAll, CalculateConjointTriad, GetQuasiSequenceOrder
import torch
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

try:
	from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
except:
	raise ImportError("Please install pip install git+https://github.com/bp-kelley/descriptastorus and pip install pandas-flavor")
from DeepPurpose.chemutils import get_mol, atom_features, bond_features, MAX_NB, ATOM_FDIM, BOND_FDIM
from subword_nmt.apply_bpe import BPE
import codecs
import pickle
import wget
from zipfile import ZipFile 
import os
import sys
import pathlib

this_dir = str(pathlib.Path(__file__).parent.absolute())

MAX_ATOM = 400
MAX_BOND = MAX_ATOM * 2

# ESPF encoding
vocab_path = f"{this_dir}/ESPF/drug_codes_chembl_freq_1500.txt"
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv(f"{this_dir}/ESPF/subword_units_map_chembl_freq_1500.csv")

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

vocab_path = f"{this_dir}/ESPF/protein_codes_uniprot_2000.txt"
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
#sub_csv = pd.read_csv(dataFolder + '/subword_units_map_protein.csv')
sub_csv = pd.read_csv(f"{this_dir}/ESPF/subword_units_map_uniprot_2000.csv")

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

from DeepPurpose.chemutils import get_mol, atom_features, bond_features, MAX_NB

def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)

def roc_curve(y_pred, y_label, figure_file, method_name):
	'''
		y_pred is a list of length n.  (0,1)
		y_label is a list of same length. 0/1
		https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py  
	'''
	import matplotlib.pyplot as plt
	from sklearn.metrics import roc_curve, auc
	from sklearn.metrics import roc_auc_score
	y_label = np.array(y_label)
	y_pred = np.array(y_pred)	
	fpr = dict()
	tpr = dict() 
	roc_auc = dict()
	fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
	roc_auc[0] = auc(fpr[0], tpr[0])
	lw = 2
	plt.plot(fpr[0], tpr[0],
         lw=lw, label= method_name + ' (area = %0.2f)' % roc_auc[0])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	fontsize = 14
	plt.xlabel('False Positive Rate', fontsize = fontsize)
	plt.ylabel('True Positive Rate', fontsize = fontsize)
	plt.title('Receiver Operating Characteristic Curve')
	plt.legend(loc="lower right")
	plt.savefig(figure_file)
	return 

def prauc_curve(y_pred, y_label, figure_file, method_name):
	'''
		y_pred is a list of length n.  (0,1)
		y_label is a list of same length. 0/1
		reference: 
			https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
	'''	
	import matplotlib.pyplot as plt
	from sklearn.metrics import precision_recall_curve, average_precision_score
	from sklearn.metrics import f1_score
	from sklearn.metrics import auc
	lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)    
#	plt.plot([0,1], [no_skill, no_skill], linestyle='--')
	plt.plot(lr_recall, lr_precision, lw = 2, label= method_name + ' (area = %0.2f)' % average_precision_score(y_label, y_pred))
	fontsize = 14
	plt.xlabel('Recall', fontsize = fontsize)
	plt.ylabel('Precision', fontsize = fontsize)
	plt.title('Precision Recall Curve')
	plt.legend()
	plt.savefig(figure_file)
	return 


def length_func(list_or_tensor):
	if type(list_or_tensor)==list:
		return len(list_or_tensor)
	return list_or_tensor.shape[0]

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

# random_fold
def create_fold(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac = test_frac, replace = False, random_state = fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]
    
    return train, val, test

# cold protein
def create_fold_setting_cold_protein(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    gene_drop = df['Target Sequence'].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values
    
    test = df[df['Target Sequence'].isin(gene_drop)]

    train_val = df[~df['Target Sequence'].isin(gene_drop)]
    
    gene_drop_val = train_val['Target Sequence'].drop_duplicates().sample(frac = val_frac/(1-test_frac), 
    																	  replace = False, 
    																	  random_state = fold_seed).values
    val = train_val[train_val['Target Sequence'].isin(gene_drop_val)]
    train = train_val[~train_val['Target Sequence'].isin(gene_drop_val)]
    
    return train, val, test

# cold drug
def create_fold_setting_cold_drug(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    drug_drop = df['SMILES'].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values
    
    test = df[df['SMILES'].isin(drug_drop)]

    train_val = df[~df['SMILES'].isin(drug_drop)]
    
    drug_drop_val = train_val['SMILES'].drop_duplicates().sample(frac = val_frac/(1-test_frac), 
    															 replace = False, 
    															 random_state = fold_seed).values
    val = train_val[train_val['SMILES'].isin(drug_drop_val)]
    train = train_val[~train_val['SMILES'].isin(drug_drop_val)]
    
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

	def __init__(self, list_IDs, labels, df, **config):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs
		self.df = df
		self.config = config

		if self.config['drug_encoding'] in ['DGL_GCN', 'DGL_NeuralFP']:
			from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
			self.node_featurizer = CanonicalAtomFeaturizer()
			self.edge_featurizer = CanonicalBondFeaturizer(self_loop = True)
			from functools import partial
			self.fc = partial(smiles_to_bigraph, add_self_loop=True)

		elif self.config['drug_encoding'] == 'DGL_AttentiveFP':
			from dgllife.utils import smiles_to_bigraph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
			self.node_featurizer = AttentiveFPAtomFeaturizer()
			self.edge_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
			from functools import partial
			self.fc = partial(smiles_to_bigraph, add_self_loop=True)

		elif self.config['drug_encoding'] in ['DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred']:
			from dgllife.utils import smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
			self.node_featurizer = PretrainAtomFeaturizer()
			self.edge_featurizer = PretrainBondFeaturizer()
			from functools import partial
			self.fc = partial(smiles_to_bigraph, add_self_loop=True)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		'Generates one sample of data'
		index = self.list_IDs[index]
		v_d = self.df.iloc[index]['drug_encoding']        
		if self.config['drug_encoding'] == 'CNN' or self.config['drug_encoding'] == 'CNN_RNN':
			v_d = drug_2_embed(v_d)
		elif self.config['drug_encoding'] in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP', 'GCN_CNN']:
			v_d = self.fc(smiles = v_d, node_featurizer = self.node_featurizer, edge_featurizer = self.edge_featurizer)
		v_p = self.df.iloc[index]['target_encoding']
		if self.config['target_encoding'] == 'CNN' or self.config['target_encoding'] == 'CNN_RNN':
			v_p = protein_2_embed(v_p)
		y = self.labels[index]
		return v_d, v_p, y

class data_process_loader_2(data.Dataset):

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

def convert_y_unit(y, from_, to_):
	array_flag = False
	if isinstance(y, (int, float)):
		y = np.array([y])
		array_flag = True
	y = y.astype(float)    
	# basis as nM
	if from_ == 'nM':
		y = y
	elif from_ == 'p':
		y = 10**(-y) / 1e-9

	if to_ == 'p':
		zero_idxs = np.where(y == 0.)[0]
		y[zero_idxs] = 1e-10
		y = -np.log10(y*1e-9)
	elif to_ == 'nM':
		y = y
        
	if array_flag:
		return y[0]
	return y

def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i, np.asarray(input_mask)

def drug2emb_encoder(x):

    max_d = 50
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
    
    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)
    '''
		the returned tuple is fed into models.transformer.forward() 
    '''

def drug2espf(x):
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
    v1 = np.zeros(len(idx2word_d),)
    v1[i1] = 1
    return v1	

def protein2espf(x):
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])
    v1 = np.zeros(len(idx2word_p),)
    v1[i1] = 1
    return v1

# '?' padding
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

from sklearn.preprocessing import OneHotEncoder
enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))

MAX_SEQ_PROTEIN = 1000
MAX_SEQ_DRUG = 100

def trans_protein(x):
	temp = list(x.upper())
	temp = [i if i in amino_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_PROTEIN:
		temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
	else:
		temp = temp [:MAX_SEQ_PROTEIN]
	return temp

def protein_2_embed(x):
	return enc_protein.transform(np.array(x).reshape(-1,1)).toarray().T

def trans_drug(x):
	temp = list(x)
	temp = [i if i in smiles_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_DRUG:
		temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
	else:
		temp = temp [:MAX_SEQ_DRUG]
	return temp

def drug_2_embed(x):
	return enc_drug.transform(np.array(x).reshape(-1,1)).toarray().T    

def save_dict(path, obj):
	with open(os.path.join(path, 'config.pkl'), 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(path):
	with open(os.path.join(path, 'config.pkl'), 'rb') as f:
		return pickle.load(f)

URLs = {
	'HIV': 'https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/hiv.zip'
	}

def obtain_compound_embedding(net, file, drug_encoding):

	if drug_encoding == 'CNN' or drug_encoding == 'CNN_RNN':
		v_d = [drug_2_embed(i) for i in file['drug_encoding'].values]
		x = np.stack(v_d)
	elif drug_encoding == 'MPNN':
		x = mpnn_collate_func(file['drug_encoding'].values)
	else:
		v_d = file['drug_encoding'].values
		x = np.stack(v_d)
	return net.model_drug(torch.FloatTensor(x))

def obtain_protein_embedding(net, file, target_encoding):

	if target_encoding == 'CNN' or target_encoding == 'CNN_RNN':
		v_d = [protein_2_embed(i) for i in file['target_encoding'].values]
		x = np.stack(v_d)
	else:
		v_d = file['target_encoding'].values
		x = np.stack(v_d)
	return net.model_protein(torch.FloatTensor(x))

## utils.smiles2mpnnfeature -> utils.mpnn_collate_func -> utils.mpnn_feature_collate_func -> encoders.MPNN.forward
def mpnn_feature_collate_func(x):
	N_atoms_scope = torch.cat([i[4] for i in x], 0)
	f_a = torch.cat([x[j][0].unsqueeze(0) for j in range(len(x))], 0)
	f_b = torch.cat([x[j][1].unsqueeze(0) for j in range(len(x))], 0)
	agraph_lst, bgraph_lst = [], []
	for j in range(len(x)):
		agraph_lst.append(x[j][2].unsqueeze(0))
		bgraph_lst.append(x[j][3].unsqueeze(0))
	agraph = torch.cat(agraph_lst, 0)
	bgraph = torch.cat(bgraph_lst, 0)
	return [f_a, f_b, agraph, bgraph, N_atoms_scope]


## utils.smiles2mpnnfeature -> utils.mpnn_collate_func -> utils.mpnn_feature_collate_func -> encoders.MPNN.forward 
def mpnn_collate_func(x):
	mpnn_feature = [i[0] for i in x]
	mpnn_feature = mpnn_feature_collate_func(mpnn_feature)
	from torch.utils.data.dataloader import default_collate
	x_remain = [list(i[1:]) for i in x]
	x_remain_collated = default_collate(x_remain)
	return [mpnn_feature] + x_remain_collated

name2ids = {
	'cnn_cnn_bindingdb': 4159715,
	'daylight_aac_bindingdb': 4159667,
	'daylight_aac_davis': 4159679,
	'daylight_aac_kiba': 4159649,
	'cnn_cnn_davis': 4159673,
	'morgan_aac_bindingdb': 4159694,
	'morgan_aac_davis': 4159706,
	'morgan_aac_kiba': 4159690,
	'morgan_cnn_bindingdb': 4159708,
	'morgan_cnn_davis': 4159705,
	'morgan_cnn_kiba': 4159676,
	'mpnn_aac_davis': 4159661,
	'mpnn_cnn_bindingdb': 4204178,
	'mpnn_cnn_davis': 4159677,
	'mpnn_cnn_kiba': 4159696,
	'transformer_cnn_bindingdb': 4159655,
	'pretrained_models': 4159682,
	'models_configs': 4159714,
	'aqsoldb_cnn_model': 4159704,
	'aqsoldb_morgan_model': 4159688,
	'aqsoldb_mpnn_model': 4159691,
	'bbb_molnet_cnn_model': 4159651,
	'bbb_molnet_mpnn_model': 4159709,
	'bbb_molnet_morgan_model': 4159703,
	'bioavailability_cnn_model': 4159663,
	'bioavailability_mpnn_model': 4159654,
	'bioavailability_morgan_model': 4159717,
	'cyp1a2_cnn_model': 4159675,
	'cyp1a2_mpnn_model': 4159671,
	'cyp1a2_morgan_model': 4159707,
	'cyp2c19_cnn_model': 4159669,
	'cyp2c19_mpnn_model': 4159687,
	'cyp2c19_morgan_model': 4159710,
	'cyp2c9_cnn_model': 4159702,
	'cyp2c9_mpnn_model': 4159686,
	'cyp2c9_morgan_model': 4159659,
	'cyp2d6_cnn_model': 4159697,
	'cyp2d6_mpnn_model': 4159674,
	'cyp2d6_morgan_model': 4159660,
	'cyp3a4_cnn_model': 4159670,
	'cyp3a4_mpnn_model': 4159678,
	'cyp3a4_morgan_model': 4159700,
	'caco2_cnn_model': 4159701,
	'caco2_mpnn_model': 4159657,
	'caco2_morgan_model': 4159662,
	'clearance_edrug3d_cnn_model': 4159699,
	'clearance_edrug3d_mpnn_model': 4159665,
	'clearance_edrug3d_morgan_model': 4159656,
	'clintox_cnn_model': 4159658,
	'clintox_mpnn_model': 4159668,
	'clintox_morgan_model': 4159713,
	'hia_cnn_model': 4159680,
	'hia_mpnn_model': 4159653,
	'hia_morgan_model': 4159711,
	'half_life_edrug3d_cnn_model': 4159716,
	'half_life_edrug3d_mpnn_model': 4159692,
	'half_life_edrug3d_morgan_model': 4159689,
	'lipo_az_cnn_model': 4159693,
	'lipo_az_mpnn_model': 4159684,
	'lipo_az_morgan_model': 4159664,
	'ppbr_cnn_model': 4159666,
	'ppbr_mpnn_model': 4159647,
	'ppbr_morgan_model': 4159685,
	'pgp_inhibitor_cnn_model': 4159646,
	'pgp_inhibitor_mpnn_model': 4159683,
	'pgp_inhibitor_morgan_model': 4159712,
	'cnn_cnn_bindingdb_ic50': 4203606,
	'daylight_aac_bindingdb_ic50': 4203604,
	'morgan_aac_bindingdb_ic50': 4203602,
	'morgan_cnn_bindingdb_ic50': 4203603,
	'mpnn_cnn_bindingdb_ic50': 4203605
 }

name2zipfilename = {
	'cnn_cnn_bindingdb': 'model_cnn_cnn_bindingdb',
	'daylight_aac_bindingdb': 'model_daylight_aac_bindingdb',
	'daylight_aac_davis': 'model_daylight_aac_davis',
	'daylight_aac_kiba': 'model_daylight_aac_kiba',
	'cnn_cnn_davis': 'model_cnn_cnn_davis',
	'morgan_aac_bindingdb': 'model_morgan_aac_bindingdb',
	'morgan_aac_davis': 'model_morgan_aac_davis',
	'morgan_aac_kiba': 'model_morgan_aac_kiba',
	'morgan_cnn_bindingdb': 'model_morgan_cnn_bindingdb',
	'morgan_cnn_davis': 'model_morgan_cnn_davis',
	'morgan_cnn_kiba': 'model_morgan_aac_kiba',
	'mpnn_aac_davis': ' model_mpnn_aac_davis',
	'mpnn_cnn_bindingdb': 'model_mpnn_cnn_bindingdb',
	'mpnn_cnn_davis': 'model_mpnn_cnn_davis',
	'mpnn_cnn_kiba': 'model_mpnn_cnn_kiba',
	'transformer_cnn_bindingdb': 'model_transformer_cnn_bindingdb',
	'pretrained_models': 'pretrained_models',
	'models_configs': 'models_configs',
	'aqsoldb_cnn_model': 'AqSolDB_CNN_model',
	'aqsoldb_mpnn_model': 'AqSolDB_MPNN_model',
	'aqsoldb_morgan_model': 'AqSolDB_Morgan_model',
	'bbb_molnet_cnn_model': 'BBB_MolNet_CNN_model',
	'bbb_molnet_mpnn_model': 'BBB_MolNet_MPNN_model',
	'bbb_molnet_morgan_model': 'BBB_MolNet_Morgan_model',
	'bioavailability_cnn_model': 'Bioavailability_CNN_model',
	'bioavailability_mpnn_model': 'Bioavailability_MPNN_model',
	'bioavailability_morgan_model': 'Bioavailability_Morgan_model',
	'cyp1a2_cnn_model': 'CYP1A2_CNN_model',
	'cyp1a2_mpnn_model': 'CYP1A2_MPNN_model',
	'cyp1a2_morgan_model': 'CYP1A2_Morgan_model',
	'cyp2c19_cnn_model': 'CYP2C19_CNN_model',
	'cyp2c19_mpnn_model': 'CYP2C19_MPNN_model',
	'cyp2c19_morgan_model': 'CYP2C19_Morgan_model',
	'cyp2c9_cnn_model': 'CYP2C9_CNN_model',
	'cyp2c9_mpnn_model': 'CYP2C9_MPNN_model',
	'cyp2c9_morgan_model': 'CYP2C9_Morgan_model',
	'cyp2d6_cnn_model': 'CYP2D6_CNN_model',
	'cyp2d6_mpnn_model': 'CYP2D6_MPNN_model',
	'cyp2d6_morgan_model': 'CYP2D6_Morgan_model',
	'cyp3a4_cnn_model': 'CYP3A4_CNN_model',
	'cyp3a4_mpnn_model': 'CYP3A4_MPNN_model',
	'cyp3a4_morgan_model': 'CYP3A4_Morgan_model',
	'caco2_cnn_model': 'Caco2_CNN_model',
	'caco2_mpnn_model': 'Caco2_MPNN_model',
	'caco2_morgan_model': 'Caco2_Morgan_model',
	'clearance_edrug3d_cnn_model': 'Clearance_eDrug3D_CNN_model',
	'clearance_edrug3d_mpnn_model': 'Clearance_eDrug3D_MPNN_model',
	'clearance_edrug3d_morgan_model': 'Clearance_eDrug3D_Morgan_model',
	'clintox_cnn_model': 'ClinTox_CNN_model',
	'clintox_mpnn_model': 'ClinTox_MPNN_model',
	'clintox_morgan_model': 'ClinTox_Morgan_model',
	'hia_cnn_model': 'HIA_CNN_model',
	'hia_mpnn_model': 'HIA_MPNN_model',
	'hia_morgan_model': 'HIA_Morgan_model',
	'half_life_edrug3d_cnn_model': 'Half_life_eDrug3D_CNN_model',
	'half_life_edrug3d_mpnn_model': 'Half_life_eDrug3D_MPNN_model',
	'half_life_edrug3d_morgan_model': 'Half_life_eDrug3D_Morgan_model',
	'lipo_az_cnn_model': 'Lipo_AZ_CNN_model',
	'lipo_az_mpnn_model': 'Lipo_AZ_MPNN_model',
	'lipo_az_morgan_model': 'Lipo_AZ_Morgan_model',
	'ppbr_cnn_model': 'PPBR_CNN_model',
	'ppbr_mpnn_model': 'PPBR_MPNN_model',
	'ppbr_morgan_model': 'PPBR_Morgan_model',
	'pgp_inhibitor_cnn_model': 'Pgp_inhibitor_CNN_model',
	'pgp_inhibitor_mpnn_model': 'Pgp_inhibitor_MPNN_model',
	'pgp_inhibitor_morgan_model': 'Pgp_inhibitor_Morgan_model',
	'cnn_cnn_bindingdb_ic50': 'cnn_cnn_bindingdb_ic50',
	'daylight_aac_bindingdb_ic50': 'daylight_aac_bindingdb_ic50',
	'morgan_aac_bindingdb_ic50': 'morgan_aac_bindingdb_ic50',
	'morgan_cnn_bindingdb_ic50': 'morgan_cnn_bindingdb_ic50',
	'mpnn_cnn_bindingdb_ic50': 'mpnn_cnn_bindingdb_ic50'
}

name2filename = {
	'cnn_cnn_bindingdb': 'model_cnn_cnn_bindingdb',
	'daylight_aac_bindingdb': 'model_daylight_aac_bindingdb',
	'daylight_aac_davis': 'model_daylight_aac_davis',
	'daylight_aac_kiba': 'model_daylight_aac_kiba',
	'cnn_cnn_davis': 'model_DeepDTA_DAVIS',
	'morgan_aac_bindingdb': 'model_morgan_aac_bindingdb',
	'morgan_aac_davis': 'model_morgan_aac_davis',
	'morgan_aac_kiba': 'model_morgan_aac_kiba',
	'morgan_cnn_bindingdb': 'model_morgan_cnn_bindingdb',
	'morgan_cnn_davis': 'model_morgan_cnn_davis',
	'morgan_cnn_kiba': 'model_morgan_aac_kiba',
	'mpnn_aac_davis': ' model_mpnn_aac_davis',
	'mpnn_cnn_bindingdb': 'model_MPNN_CNN',
	'mpnn_cnn_davis': 'model_mpnn_cnn_davis',
	'mpnn_cnn_kiba': 'model_mpnn_cnn_kiba',
	'transformer_cnn_bindingdb': 'model_transformer_cnn_bindingdb',
	'pretrained_models': 'DeepPurpose_BindingDB',
	'models_configs': 'models_configs',
	'aqsoldb_cnn_model': 'AqSolDB_CNN_model',
	'aqsoldb_mpnn_model': 'AqSolDB_MPNN_model',
	'aqsoldb_morgan_model': 'AqSolDB_Morgan_model',
	'bbb_molnet_cnn_model': 'BBB_MolNet_CNN_model',
	'bbb_molnet_mpnn_model': 'BBB_MolNet_MPNN_model',
	'bbb_molnet_morgan_model': 'BBB_MolNet_Morgan_model',
	'bioavailability_cnn_model': 'Bioavailability_CNN_model',
	'bioavailability_mpnn_model': 'Bioavailability_MPNN_model',
	'bioavailability_morgan_model': 'Bioavailability_Morgan_model',
	'cyp1a2_cnn_model': 'CYP1A2_CNN_model',
	'cyp1a2_mpnn_model': 'CYP1A2_MPNN_model',
	'cyp1a2_morgan_model': 'CYP1A2_Morgan_model',
	'cyp2c19_cnn_model': 'CYP2C19_CNN_model',
	'cyp2c19_mpnn_model': 'CYP2C19_MPNN_model',
	'cyp2c19_morgan_model': 'CYP2C19_Morgan_model',
	'cyp2c9_cnn_model': 'CYP2C9_CNN_model',
	'cyp2c9_mpnn_model': 'CYP2C9_MPNN_model',
	'cyp2c9_morgan_model': 'CYP2C9_Morgan_model',
	'cyp2d6_cnn_model': 'CYP2D6_CNN_model',
	'cyp2d6_mpnn_model': 'CYP2D6_MPNN_model',
	'cyp2d6_morgan_model': 'CYP2D6_Morgan_model',
	'cyp3a4_cnn_model': 'CYP3A4_CNN_model',
	'cyp3a4_mpnn_model': 'CYP3A4_MPNN_model',
	'cyp3a4_morgan_model': 'CYP3A4_Morgan_model',
	'caco2_cnn_model': 'Caco2_CNN_model',
	'caco2_mpnn_model': 'Caco2_MPNN_model',
	'caco2_morgan_model': 'Caco2_Morgan_model',
	'clearance_edrug3d_cnn_model': 'Clearance_eDrug3D_CNN_model',
	'clearance_edrug3d_mpnn_model': 'Clearance_eDrug3D_MPNN_model',
	'clearance_edrug3d_morgan_model': 'Clearance_eDrug3D_Morgan_model',
	'clintox_cnn_model': 'ClinTox_CNN_model',
	'clintox_mpnn_model': 'ClinTox_MPNN_model',
	'clintox_morgan_model': 'ClinTox_Morgan_model',
	'hia_cnn_model': 'HIA_CNN_model',
	'hia_mpnn_model': 'HIA_MPNN_model',
	'hia_morgan_model': 'HIA_Morgan_model',
	'half_life_edrug3d_cnn_model': 'Half_life_eDrug3D_CNN_model',
	'half_life_edrug3d_mpnn_model': 'Half_life_eDrug3D_MPNN_model',
	'half_life_edrug3d_morgan_model': 'Half_life_eDrug3D_Morgan_model',
	'lipo_az_cnn_model': 'Lipo_AZ_CNN_model',
	'lipo_az_mpnn_model': 'Lipo_AZ_MPNN_model',
	'lipo_az_morgan_model': 'Lipo_AZ_Morgan_model',
	'ppbr_cnn_model': 'PPBR_CNN_model',
	'ppbr_mpnn_model': 'PPBR_MPNN_model',
	'ppbr_morgan_model': 'PPBR_Morgan_model',
	'pgp_inhibitor_cnn_model': 'Pgp_inhibitor_CNN_model',
	'pgp_inhibitor_mpnn_model': 'Pgp_inhibitor_MPNN_model',
	'pgp_inhibitor_morgan_model': 'Pgp_inhibitor_Morgan_model',
	'cnn_cnn_bindingdb_ic50': 'cnn_cnn_bindingdb_ic50',
	'daylight_aac_bindingdb_ic50': 'daylight_aac_bindingdb_ic50',
	'morgan_aac_bindingdb_ic50': 'morgan_aac_bindingdb_ic50',
	'morgan_cnn_bindingdb_ic50': 'morgan_cnn_bindingdb_ic50',
	'mpnn_cnn_bindingdb_ic50': 'mpnn_cnn_bindingdb_ic50'
}

def download_unzip(name, path, file_name):
	if not os.path.exists(path):
		os.mkdir(path)

	if os.path.exists(os.path.join(path, file_name)):
		print('Dataset already downloaded in the local system...', flush = True, file = sys.stderr)
	else:
		print('Download zip file...', flush = True, file = sys.stderr)
		url = URLs[name]
		saved_path = wget.download(url, path)

		print('Extract zip file...', flush = True, file = sys.stderr)
		with ZipFile(saved_path, 'r') as zip: 
		    zip.extractall(path = path) 

def download_pretrained_model(model_name, save_dir = './save_folder'):
	
	if model_name.lower() in list(name2ids.keys()):
		server_path = 'https://dataverse.harvard.edu/api/access/datafile/'
		url = server_path + str(name2ids[model_name.lower()])
	else:
		raise Exception("Given name not a pretrained model. The full list is in the Github README https://github.com/kexinhuang12345/DeepPurpose/blob/master/README.md#pretrained-models")

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	if not os.path.exists(os.path.join(save_dir, 'pretrained_models')):
		os.mkdir(os.path.join(save_dir, 'pretrained_models'))

	pretrained_dir = os.path.join(save_dir, 'pretrained_models')
	downloaded_path = os.path.join(pretrained_dir, name2zipfilename[model_name.lower()] + '.zip')

	if os.path.exists(os.path.join(pretrained_dir, name2filename[model_name.lower()])):
		print('Dataset already downloaded in the local system...')
	else:
		download_url(url, downloaded_path)
		print('Downloading finished... Beginning to extract zip file...')
		with ZipFile(downloaded_path, 'r') as zip: 
			zip.extractall(path = pretrained_dir)
		print('pretrained model Successfully Downloaded...')
	
	pretrained_dir = os.path.join(pretrained_dir, name2filename[model_name.lower()])
	
	return pretrained_dir
	
import requests 

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def download_pretrained_model_S3(model_name, save_dir = './save_folder'):
	print('Beginning Downloading' + model_name + ' Model...')
	url = 'https://deeppurpose.s3.amazonaws.com/' + model_name + '.zip'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	if not os.path.exists(os.path.join(save_dir, 'pretrained_model')):
		os.mkdir(os.path.join(save_dir, 'pretrained_model'))

	pretrained_dir = os.path.join(save_dir, 'pretrained_model')
	
	if not os.path.exists(os.path.join(pretrained_dir, model_name)):
		pretrained_dir_ = wget.download(url, pretrained_dir)
		print('Downloading finished... Beginning to extract zip file...')
		with ZipFile(pretrained_dir_, 'r') as zip: 
			zip.extractall(path = pretrained_dir)
		print('pretrained model Successfully Downloaded...')
	
	pretrained_dir = os.path.join(pretrained_dir, model_name)
	return pretrained_dir
