import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from models.lib.weight_util import weights_init
from torch.nn.parameter import Parameter
from models.layers.graph_convolution_layers import GraphConvolutionLayers_DGCNN

class GCND(nn.Module):
	def __init__(self, config, dataset_features, regression=False):
		super(GCND, self).__init__()
		self.regression = regression
		self.config = config
		self.dataset_features = dataset_features

		# Initialise Graph Convolution Layers
		if '-' in str(self.config["convolution_layers_size"]):
			self.config["convolution_layers_size"] = \
				list(map(int, self.config["convolution_layers_size"].split('-')))
		else:
			self.config["convolution_layers_size"] = [int(self.config["convolution_layers_size"])]

		self.graph_convolution = GraphConvolutionLayers_DGCNN(
			latent_dim=self.config["convolution_layers_size"],
			input_dim=dataset_features["feat_dim"] + dataset_features["attr_dim"] + dataset_features["edge_feat_dim"],
			concat_tensors=False)

		self.weight = Parameter(torch.FloatTensor(config["convolution_layers_size"][-1],
												  dataset_features["num_class"]))

		weights_init(self)

	def forward(self, node_feat, adjacency_matrix, subg, batch_graph):
		graph_sizes = [batch_graph[i].number_of_nodes for i in range(len(batch_graph))]
		output_matrix = self.graph_convolution(node_feat, adjacency_matrix, batch_graph)

		batch_logits = torch.zeros(len(graph_sizes), self.dataset_features["num_class"])

		accum_count=0
		for i in range(subg):
			to_pool = output_matrix[accum_count:accum_count+graph_sizes[i]]
			average_pooling = to_pool.mean(0, keepdim=True)
			pool_out = average_pooling.mm(self.weight)
			batch_logits[i] = pool_out

		return batch_logits

	def output_features(self, batch_graph):
		embed = self.graph_convolution(batch_graph)
		return embed, labels

	def loss(self, logits, labels):
		return F.nll_loss(logits, labels)