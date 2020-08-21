import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from models.lib.weight_util import weights_init
from models.layers.graph_convolution_layer import GraphConvolutionLayer_DGCNN_Dense
from models.layers.graph_convolution_layers import GraphConvolutionLayers_DGCNN_Dense
from models.layers.dense_layers import DenseLayers

class DiffPoolD(nn.Module):
	def __init__(self, config, dataset_features, regression=False):
		super(DiffPoolD, self).__init__()
		self.regression = regression
		self.config = config
		self.dataset_features = dataset_features
		self.concat_tensors = config["concat_tensors"]
		self.num_pooling = config["number_of_pooling"]
		self.assign_ratio = config["assign_ratio"]
		self.input_dim = dataset_features["feat_dim"] + dataset_features["attr_dim"] + dataset_features["edge_feat_dim"]

		# Embedding Tensor
		if '-' in str(self.config["convolution_layers_size"]):
			self.config["convolution_layers_size"] = \
				list(map(int, self.config["convolution_layers_size"].split('-')))
		else:
			self.config["convolution_layers_size"] = [int(self.config["convolution_layers_size"])]

		self.graph_convolution = GraphConvolutionLayers_DGCNN_Dense(
			latent_dim=self.config["convolution_layers_size"],
			input_dim=self.input_dim,
			concat_tensors=self.concat_tensors)

		if self.concat_tensors is True:
			self.pred_input_dim = sum(self.config["convolution_layers_size"])
		else:
			self.pred_input_dim = self.config["convolution_layers_size"][-1]

		# DiffPoolD Layers (Assignment)
		self.conv_modules = nn.ModuleList()
		self.assign_modules = nn.ModuleList()
		self.assign_pred_modules = nn.ModuleList()

		# Initialise first assign dimension
		assign_input_dim = self.input_dim
		assign_dim = int(self.dataset_features["max_num_nodes"] * self.config["assign_ratio"])

		for stack in range(self.num_pooling):
			# GNN Pool
			self.conv_modules.append(GraphConvolutionLayers_DGCNN_Dense(
				latent_dim=self.config["convolution_layers_size"],
				input_dim=self.pred_input_dim,
				concat_tensors=self.concat_tensors))

			# GNN Assign
			self.assign_modules.append(GraphConvolutionLayers_DGCNN_Dense(
				latent_dim=self.config["convolution_layers_size"] + [assign_dim],
				input_dim=assign_input_dim,
				concat_tensors=self.concat_tensors))

			if self.concat_tensors is True:
				assign_pred_input_dim = sum(self.config["convolution_layers_size"]) + assign_dim
			else:
				assign_pred_input_dim = assign_dim
			self.assign_pred_modules.append(DenseLayers(assign_pred_input_dim, assign_dim, []))

			# For next pooling stack
			assign_input_dim = self.pred_input_dim
			assign_dim = int(assign_dim * self.assign_ratio)

		# Prediction Layers
		self.config["pred_hidden_layers"] = \
			list(map(int, self.config["pred_hidden_layers"].split('-')))

		if self.concat_tensors is True:
			self.prediction_model = DenseLayers(self.pred_input_dim * (self.num_pooling+1), dataset_features["num_class"],
												self.config["pred_hidden_layers"])
		else:
			self.prediction_model = DenseLayers(self.pred_input_dim, dataset_features["num_class"],
												self.config["pred_hidden_layers"])

		# Initialise weights
		weights_init(self)

	def forward(self, node_feat, adjacency_matrix, subg_size, batch_graph):
		adjacency_matrix = adjacency_matrix.to_dense()
		self.input_adj = adjacency_matrix

		# Store assignment matrix
		self.cur_assign_tensor_list = []

		node_feat_a = node_feat
		out_all = []

		# Embedding Tensor
		embedding_tensor = self.graph_convolution(node_feat, adjacency_matrix, batch_graph)
		out, _ = torch.max(embedding_tensor, dim=0, keepdim=True)
		out_all.append(out)

		for stack in range(self.num_pooling):
			assign_tensor = self.assign_modules[stack](node_feat_a, adjacency_matrix, batch_graph)
			assign_tensor = nn.Softmax(dim	=-1)(self.assign_pred_modules[stack](assign_tensor))
			self.cur_assign_tensor_list.append(assign_tensor)

			# update pooled features and adj matrix
			node_feat = torch.matmul(torch.transpose(assign_tensor, 0, 1), embedding_tensor)
			adjacency_matrix = torch.transpose(assign_tensor, 0, 1) @ adjacency_matrix @ assign_tensor
			node_feat_a = node_feat
			embedding_tensor = self.conv_modules[stack](node_feat, adjacency_matrix, batch_graph)

			out, _ = torch.max(embedding_tensor, dim=0, keepdim=True)
			out_all.append(out)

		if self.concat_tensors:
			output = torch.cat(out_all, dim=1)
		else:
			output = out

		pred = self.prediction_model(output)

		return pred

	def loss(self, logits, labels):
		loss = F.cross_entropy(logits, labels, reduction='mean')
		return loss
