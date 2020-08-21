import torch.nn as nn
import torch.nn.functional as F
import math

from models.lib.weight_util import weights_init
from models.layers.graph_convolution_layers import GraphConvolutionLayers_DGCNN
from models.layers.mlp_layers import MLPClassifier
from models.layers.sortpooling import SortPooling

class DGCNN(nn.Module):
	def __init__(self, config, dataset_features):
		super(DGCNN, self).__init__()
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
			concat_tensors=True,
			dropout=self.config["convolution_dropout"])

		# Initialise Sortpooling Layer
		if 1 >= self.config["sortpooling_k"] > 0:
			num_nodes_list = sorted(dataset_features['graph_sizes_list'])
			self.config["sortpooling_k"] = num_nodes_list[
				int(math.ceil(self.config["sortpooling_k"] * len(num_nodes_list))) - 1]
			self.config["sortpooling_k"] = max(10, self.config["sortpooling_k"])
			print('k used in SortPooling is: ' + str(self.config["sortpooling_k"]))
		else:
			print('Invalid sortpooling_k, it needs to be between 0 and 1')
			exit()

		self.sort_pool = SortPooling(self.config["sortpooling_k"],
									 sum(config["convolution_layers_size"]))

		# Initialise MLP Classification Layers
		self.mlp = MLPClassifier(
			output_dim=self.config["FP_len"],
			hidden_size=self.config["n_hidden"],
			num_class=self.dataset_features["num_class"],
			dropout=self.config["pred_dropout"],
			latent_dim=self.config["convolution_layers_size"],
			k=self.config["sortpooling_k"])

		weights_init(self)

	def forward(self, node_feat, adjacency_matrix, subg_size, batch_graph):
		graph_sizes = [batch_graph[i].number_of_nodes for i in range(len(batch_graph))]

		output_matrix = self.graph_convolution(node_feat, adjacency_matrix, batch_graph)
		embed, _ = self.sort_pool(output_matrix, subg_size, graph_sizes)
		return self.mlp(embed, graph_sizes)

	def loss(self, logits, labels):
		return F.nll_loss(logits, labels)