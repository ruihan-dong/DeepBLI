import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import SequentialSampler
from torch import nn 


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, log_loss
from sklearn.metrics import recall_score, precision_score, matthews_corrcoef
import pickle 
torch.manual_seed(2)
np.random.seed(3)
import copy
from prettytable import PrettyTable

import os

from DeepPurpose.utils import *     
from DeepPurpose.encoders import *
import utils_multi as um

from torch.utils.tensorboard import SummaryWriter


class Classifier(nn.Sequential):
	def __init__(self, model_drug_1, model_protein_1, model_drug_2, model_protein_2, **config):
		super(Classifier, self).__init__()
		self.input_dim_drug = config['hidden_dim_drug']
		self.input_dim_protein = config['hidden_dim_protein']

		self.model_drug_1 = model_drug_1
		self.model_protein_1 = model_protein_1
		self.model_drug_2 = model_drug_2
		self.model_protein_2 = model_protein_2

		self.dropout = nn.Dropout(0.1)

		self.hidden_dims = config['cls_hidden_dims']
		layer_size = len(self.hidden_dims) + 1
		dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [1]

		# linear for attention
		self.attention_drug = nn.Linear(self.input_dim_drug, self.input_dim_protein)
		self.attention_protein = nn.Linear(self.input_dim_protein, self.input_dim_drug)

		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])
	
	def coAttention(self, v_D, v_P):
		# co-attention
		att_drug = self.attention_drug(v_D)
		att_protein = self.attention_protein(v_P)
		att_output_d = F.softmax(att_drug, 1)
		att_output_p = F.softmax(att_protein, 1)
		v_D = torch.mul(v_D, att_output_p)
		v_P = torch.mul(v_P, att_output_d)
		v_f = torch.cat((v_D, v_P), 1)
		return v_f
	
	def forward(self, v_D1, v_P1, v_D2, v_P2):
		# each encoding
		v_D1 = self.model_drug_1(v_D1)
		v_P1 = self.model_protein_1(v_P1)
		v_D2 = self.model_drug_2(v_D2)
		v_P2 = self.model_protein_2(v_P2)

		v_f1 = self.coAttention(v_D1, v_P1)
		v_f2 = self.coAttention(v_D2, v_P2)
		v_f = torch.mul(v_f1, v_f2)

		# classify
		for i, l in enumerate(self.predictor):
			if i==(len(self.predictor)-1):
				v_f = l(v_f)
			else:
				v_f = F.relu(self.dropout(l(v_f)))
		return v_f

def model_initialize(**config):
	model = DBTA(**config)
	return model

def model_pretrained(path_dir = None, model = None):
	if model is not None:
		path_dir = download_pretrained_model(model)
	config = load_dict(path_dir)
	model = DBTA(**config)
	model.load_pretrained(path_dir + '/model.pt')    
	return model

# collate function has been modified for GCN encoder
def dgl_collate_func(x):
	d1, p1, d2, p2, y = zip(*x)
	import dgl
	d1 = dgl.batch(d1)
	d2 = dgl.batch(d2)
	# x_remain = [list(i[1:]) for i in x]
	from torch.utils.data.dataloader import default_collate
	p1_collated = default_collate(p1)
	p2_collated = default_collate(p2)
	y_collated = default_collate(y)
	return [d1] + [p1_collated] + [d2] + [p2_collated] + [y_collated]

class DBTA:

	def __init__(self, **config):
		drug_encoding_1 = config['drug_encoding_1']
		target_encoding_1 = config['target_encoding_1']
		drug_encoding_2 = config['drug_encoding_2']
		target_encoding_2 = config['target_encoding_2']

		self.model_drug_1 = DGL_GCN(in_feats = 74, 
									hidden_feats = [config['gnn_hid_dim_drug']] * config['gnn_num_layers'], 
									activation = [config['gnn_activation']] * config['gnn_num_layers'], 
									predictor_dim = config['hidden_dim_drug'])
		
		self.model_drug_2 = DGL_AttentiveFP(node_feat_size = 39, 
											edge_feat_size = 11,  
											num_layers = config['gnn_num_layers'], 
											num_timesteps = config['attentivefp_num_timesteps'], 
											graph_feat_size = config['gnn_hid_dim_drug'], 
											predictor_dim = config['hidden_dim_drug'])

		self.model_protein_1 = CNN('protein', **config)
		self.model_protein_2 = transformer('protein', **config)

		self.model = Classifier(self.model_drug_1, self.model_protein_1, self.model_drug_2, self.model_protein_2, **config)
		self.config = config

		if 'cuda_id' in self.config:
			if self.config['cuda_id'] is None:
				self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			else:
				self.device = torch.device('cuda:' + str(self.config['cuda_id']) if torch.cuda.is_available() else 'cpu')
		else:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		self.drug_encoding_1 = drug_encoding_1
		self.target_encoding_1 = target_encoding_1
		self.drug_encoding_2 = drug_encoding_2
		self.target_encoding_2 = target_encoding_2

		self.result_folder = config['result_folder']
		if not os.path.exists(self.result_folder):
			os.mkdir(self.result_folder)            
		self.binary = False
		if 'num_workers' not in self.config.keys():
			self.config['num_workers'] = 0
		if 'decay' not in self.config.keys():
			self.config['decay'] = 0

	def test_(self, data_generator, model, repurposing_mode = False, test = False):
		y_pred = []
		y_label = []
		model.eval()
		for i, (v_d1, v_p1, v_d2, v_p2, label) in enumerate(data_generator):
			v_d1 = v_d1
			v_d2 = v_d2
			v_p1 = v_p1.float().to(self.device)
			v_p2 = v_p2
			score = self.model(v_d1, v_p1, v_d2, v_p2)

			m = torch.nn.Sigmoid()
			logits = torch.squeeze(m(score)).detach().cpu().numpy()

			label_ids = label.to('cpu').numpy()
			y_label = y_label + label_ids.flatten().tolist()
			y_pred = y_pred + logits.flatten().tolist()
			outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
		
		model.train()
		if repurposing_mode:
			return y_pred, outputs
		if test:
			roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
			plt.figure(0)
			roc_curve(y_pred, y_label, roc_auc_file, self.drug_encoding_1 + '_' + self.target_encoding_1)
			plt.figure(1)
			pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
			prauc_curve(y_pred, y_label, pr_auc_file, self.drug_encoding_1 + '_' + self.target_encoding_1)

		return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), accuracy_score(y_label, outputs), f1_score(y_label, outputs), log_loss(y_label, outputs), y_pred, recall_score(y_label, outputs), precision_score(y_label, outputs), matthews_corrcoef(y_label, outputs)

	def train(self, train1, val1, test1, train2, val2 = None, test2 = None, verbose = True):
		if len(train1.Label.unique()) == 2:
			self.binary = True
			self.config['binary'] = True

		lr = self.config['LR']
		decay = self.config['decay']
		BATCH_SIZE = self.config['batch_size']
		train_epoch = self.config['train_epoch']
		if 'test_every_X_epoch' in self.config.keys():
			test_every_X_epoch = self.config['test_every_X_epoch']
		else:     
			test_every_X_epoch = 40
		loss_history = []

		self.model = self.model.to(self.device)

		# support multiple GPUs
		if torch.cuda.device_count() > 1:
			if verbose:
				print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
			self.model = nn.DataParallel(self.model, dim = 0)
		elif torch.cuda.device_count() == 1:
			if verbose:
				print("Let's use " + str(torch.cuda.device_count()) + " GPU!")
		else:
			if verbose:
				print("Let's use CPU/s!")
		opt = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = decay)
		if verbose:
			print('--- Data Preparation ---')

		params = {'batch_size': BATCH_SIZE,
	    		'shuffle': True,
	    		'num_workers': self.config['num_workers'],
	    		'drop_last': False}
		params['collate_fn'] = dgl_collate_func

		training_generator = data.DataLoader(um.data_process_loader(train1.index.values, train1.Label.values, train1, 
																train2.index.values, train2, **self.config), **params)

		if val1 is not None:
			validation_generator = data.DataLoader(um.data_process_loader(val1.index.values, val1.Label.values, val1,
																		val2.index.values, val2, **self.config), **params)

		if test1 is not None:
			info = um.data_process_loader(test1.index.values, test1.Label.values, test1, test2.index.values, test2,**self.config)
			params_test = {'batch_size': BATCH_SIZE,
					'shuffle': False,
					'num_workers': self.config['num_workers'],
					'drop_last': False,
					'sampler':SequentialSampler(info)}
			params_test['collate_fn'] = dgl_collate_func
			testing_generator = data.DataLoader(um.data_process_loader(test1.index.values, test1.Label.values, test1,
																	test2.index.values, test2, **self.config), **params_test)

		# early stopping
		if self.binary:
			max_auc = 0
		else:
			max_MSE = 10000
		model_max = copy.deepcopy(self.model)

		valid_metric_record = []
		valid_metric_header = ["# epoch"] 
		valid_metric_header.extend(["AUROC", "AUPRC", "ACC", "F1", "Recall", "Precision", "MCC"])
		table = PrettyTable(valid_metric_header)
		float2str = lambda x:'%0.4f'%x
		
		if verbose:
			print('--- Go for Training ---')
		writer = SummaryWriter()
		t_start = time() 
		iteration_loss = 0
		for epo in range(train_epoch):
			for i, (v_d1, v_p1, v_d2, v_p2, label) in enumerate(training_generator):
				v_d1 = v_d1
				v_d2 = v_d2
				v_p1 = v_p1.float().to(self.device)
				v_p2 = v_p2

				score = self.model(v_d1, v_p1, v_d2, v_p2)
				label = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)

				# Training loss function
				'''
				loss_fct = torch.nn.BCELoss()
				m = torch.nn.Sigmoid()
				n = torch.squeeze(m(score), 1)
				'''
				loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([0.3]).float().to(self.device))
				n = torch.squeeze(score, 1)
				
				loss = loss_fct(n, label)
				
				loss_history.append(loss.item())
				writer.add_scalar("Loss/train", loss.item(), iteration_loss)
				iteration_loss += 1

				opt.zero_grad()
				loss.backward()
				opt.step()

				if verbose:
					if (i % 100 == 0):
						t_now = time()
						print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + \
							' with loss ' + str(loss.cpu().detach().numpy())[:7] +\
							". Total time " + str(int(t_now - t_start)/3600)[:7] + " hours") 
						### record total run time
						
			if val1 is not None:
				##### validate, select the best model up to now 
				with torch.set_grad_enabled(False):
					## binary: ROC-AUC, PR-AUC, ACC, F1, cross-entropy loss, recall, precision, MCC
					auc, auprc, acc, f1, loss, logits, recall, precision, mcc = self.test_(validation_generator, self.model)
					lst = ["epoch " + str(epo)] + list(map(float2str,[auc, auprc, acc, f1, recall, precision, mcc]))
					valid_metric_record.append(lst)
					if auc > max_auc:
						model_max = copy.deepcopy(self.model)
						max_auc = auc   
					if verbose:
						print('Validation at Epoch '+ str(epo + 1) + ', AUROC: ' + str(auc)[:7] + \
							' , AUPRC: ' + str(auprc)[:7] + ' , Acc: '+str(acc)[:7] +' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
							str(loss)[:7] + ' , Recall: ' + str(recall)[:7] + ' , Precision: '+str(precision)[:7] +' , MCC: '+str(mcc)[:7])
				table.add_row(lst)
			else:
				model_max = copy.deepcopy(self.model)

		# load early stopped model
		self.model = model_max

		if val1 is not None:
			#### after training 
			prettytable_file = os.path.join(self.result_folder, "valid_markdowntable.txt")
			with open(prettytable_file, 'w') as fp:
				fp.write(table.get_string())
		
		if test1 is not None:
			if verbose:
				print('--- Go for Testing ---')
			# with torch.no_grad():
				auc, auprc, acc, f1, loss, logits, recall, precision, mcc = self.test_(testing_generator, model_max, test = True)
			test_table = PrettyTable(["AUROC", "AUPRC", "ACC", "F1", "Recall", "Precision", "MCC"])
			test_table.add_row(list(map(float2str, [auc, auprc, acc, f1, recall, precision, mcc])))
			# epo = 0
			if verbose:
				print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: ' + str(auc)[:7] + \
					' , AUPRC: ' + str(auprc)[:7] + ' , Acc: '+str(acc)[:7] +' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
					str(loss)[:7] + ' , Recall: ' + str(recall)[:7] + ' , Precision: '+str(precision)[:7] +' , MCC: '+str(mcc)[:7])				

			np.save(os.path.join(self.result_folder, str(self.drug_encoding_1) + '_' + str(self.target_encoding_1) 
				     + '_logits.npy'), np.array(logits))                
	
			######### learning record ###########

			### 1. test results
			prettytable_file = os.path.join(self.result_folder, "test_markdowntable.txt")
			with open(prettytable_file, 'w') as fp:
				fp.write(test_table.get_string())

		### 2. learning curve 
		fontsize = 16
		iter_num = list(range(1,len(loss_history)+1))
		plt.figure(3)
		plt.plot(iter_num, loss_history, "bo-")
		plt.xlabel("iteration", fontsize = fontsize)
		plt.ylabel("loss value", fontsize = fontsize)
		pkl_file = os.path.join(self.result_folder, "loss_curve_iter.pkl")
		with open(pkl_file, 'wb') as pck:
			pickle.dump(loss_history, pck)

		fig_file = os.path.join(self.result_folder, "loss_curve.png")
		plt.savefig(fig_file)
		if verbose:
			print('--- Training Finished ---')
			writer.flush()
			writer.close()
          

	def predict(self, df_data_1, df_data_2):
		'''
			utils.data_process_repurpose_virtual_screening 
			pd.DataFrame
		'''
		print('predicting...')
		info = um.data_process_loader(df_data_1.index.values, df_data_1.Label.values, df_data_1, 
									df_data_2.index.values, df_data_2, **self.config)
		self.model.to(self.device)
		params = {'batch_size': self.config['batch_size'],
				'shuffle': False,
				'num_workers': self.config['num_workers'],
				'drop_last': False,
				'sampler':SequentialSampler(info)}
		params['collate_fn'] = dgl_collate_func

		generator = data.DataLoader(info, **params)

		score, output = self.test_(generator, self.model, repurposing_mode = True)
		return score, output

	def save_model(self, path_dir):
		if not os.path.exists(path_dir):
			os.makedirs(path_dir)
		torch.save(self.model.state_dict(), path_dir + '/model.pt')
		save_dict(path_dir, self.config)
	
	def load_pretrained(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

		state_dict = torch.load(path, map_location = torch.device('cpu'))
		# to support training from multi-gpus data-parallel:
        
		if next(iter(state_dict))[:7] == 'module.':
			# the pretrained model is from data-parallel module
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				name = k[7:] # remove `module.`
				new_state_dict[name] = v
			state_dict = new_state_dict

		self.model.load_state_dict(state_dict)

		self.binary = self.config['binary']