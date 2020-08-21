import torch
import torch.nn as nn
from models.layers.graph_convolution_layer import GraphConvolutionLayer_GCN, GraphConvolutionLayer_DGCNN, GraphConvolutionLayer_DGCNN_Dense, GraphConvolutionLayer_GraphSAGE

# Torch nn module: Graph Convolution Layers using GCN implementation
class GraphConvolutionLayers_GCN(nn.Module):
	'''
		Graph Convolution layers
	'''
	def __init__(self,
		input_dim,
		latent_dim=[128, 256, 512],
		concat_tensors=False,
		dropout=0.0):

		# Intialise settings
		super(GraphConvolutionLayers_GCN, self).__init__()
		self.latent_dim = latent_dim
		self.total_latent_dim = sum(latent_dim)
		self.concat_tensors = concat_tensors

		# Create convolution Layers as module list
		self.conv_layers = nn.ModuleList()

		# First layer takes in the node feature X
		self.conv_layers.append(GraphConvolutionLayer_GCN(input_dim,
													  latent_dim[0],
													  dropout))

		# Following layers take latent dim of previous convolution layer as input
		for i in range(1, len(latent_dim)):
			self.conv_layers.append(GraphConvolutionLayer_GCN(latent_dim[i-1], latent_dim[i], dropout))

	def forward(self, node_feat, adjacency_matrix, batch_graph):
		node_degs = [torch.Tensor(batch_graph[i].node_degrees) + 1 for i in range(len(batch_graph))]
		node_degs = torch.cat(node_degs).unsqueeze(1)

		# Graph Convolution Layers Forward
		lv = 0
		output_matrix = node_feat
		cat_output_matrix = []
		while lv < len(self.latent_dim):
			output_matrix = self.conv_layers[lv](output_matrix, adjacency_matrix, node_degs)
			cat_output_matrix.append(output_matrix)
			lv += 1

		if self.concat_tensors:
			return torch.cat(cat_output_matrix, 1)
		else:
			return output_matrix

# Torch nn module: Graph Convolution Layers using DGCNN implementation
class GraphConvolutionLayers_DGCNN(nn.Module):
	'''
		Graph Convolution layers using sparse adjacency matrix
	'''
	def __init__(self,
		input_dim,
		latent_dim=[32, 32, 32, 1],
		concat_tensors=False,
		dropout=0.0):

		# Intialise settings
		super(GraphConvolutionLayers_DGCNN, self).__init__()
		self.latent_dim = latent_dim
		self.total_latent_dim = sum(latent_dim)
		self.concat_tensors = concat_tensors

		# Create convolution Layers as module list
		self.conv_layers = nn.ModuleList()

		# First layer takes in the node feature X
		self.conv_layers.append(GraphConvolutionLayer_DGCNN(input_dim,
													  latent_dim[0],
													  dropout))

		# Following layers take latent dim of previous convolution layer as input
		for i in range(1, len(latent_dim)):
			self.conv_layers.append(GraphConvolutionLayer_DGCNN(latent_dim[i-1], latent_dim[i], dropout))

	def forward(self, node_feat, adjacency_matrix, batch_graph):
		node_degs = torch.transpose(torch.sum(adjacency_matrix.to_dense(), dim=0, keepdim=True).add(1), 0, 1)

		# Graph Convolution Layers Forward
		lv = 0
		output_matrix = node_feat
		cat_output_matrix= []
		while lv < len(self.latent_dim):
			output_matrix = self.conv_layers[lv](output_matrix, adjacency_matrix, node_degs)
			cat_output_matrix.append(output_matrix)
			lv += 1

		if self.concat_tensors:
			return torch.cat(cat_output_matrix, 1)
		else:
			return output_matrix

class GraphConvolutionLayers_DGCNN_Dense(nn.Module):
	'''
		Graph Convolution layers using dense adjacency matrix
	'''
	def __init__(self,
		input_dim,
		latent_dim=[32, 32, 32, 1],
		concat_tensors=False,
		dropout=0.0):

		# Intialise settings
		super(GraphConvolutionLayers_DGCNN_Dense, self).__init__()
		self.latent_dim = latent_dim
		self.total_latent_dim = sum(latent_dim)
		self.concat_tensors = concat_tensors

		# Create convolution Layers as module list
		self.conv_layers = nn.ModuleList()

		# First layer takes in the node feature X
		self.conv_layers.append(GraphConvolutionLayer_DGCNN_Dense(input_dim,
													  latent_dim[0],
													  dropout))

		# Following layers take latent dim of previous convolution layer as input
		for i in range(1, len(latent_dim)):
			self.conv_layers.append(GraphConvolutionLayer_DGCNN_Dense(latent_dim[i-1], latent_dim[i], dropout))

	def forward(self, node_feat, adjacency_matrix, batch_graph):
		node_degs = torch.transpose(torch.sum(adjacency_matrix, dim=0, keepdim=True).add(1), 0, 1)

		# Graph Convolution Layers Forward
		lv = 0
		output_matrix = node_feat
		cat_output_matrix= []
		while lv < len(self.latent_dim):
			output_matrix = self.conv_layers[lv](output_matrix, adjacency_matrix, node_degs)
			cat_output_matrix.append(output_matrix)
			lv += 1

		if self.concat_tensors:
			return torch.cat(cat_output_matrix, 1)
		else:
			return output_matrix

# Torch nn module: Graph Convolution Layers using GraphSAGE implementation
class GraphConvolutionLayers_GraphSAGE(nn.Module):
	'''
		Graph Convolution layers
	'''
	def __init__(self,
		input_dim,
		latent_dim=[64, 64, 64],
		concat_tensors=False,
		dropout=0.0):

		# Intialise settings
		super(GraphConvolutionLayers_GraphSAGE, self).__init__()
		self.latent_dim = latent_dim
		self.total_latent_dim = sum(latent_dim)
		self.concat_tensors = concat_tensors

		# Create convolution Layers as module list
		self.conv_layers = nn.ModuleList()

		# First layer takes in the node feature X
		self.conv_layers.append(GraphConvolutionLayer_GraphSAGE(input_dim,
													  latent_dim[0], add_self = not self.concat_tensors,
													  dropout = dropout))

		# Following layers take latent dim of previous convolution layer as input
		for i in range(1, len(latent_dim)):
			self.conv_layers.append(GraphConvolutionLayer_GraphSAGE(latent_dim[i-1], latent_dim[i], dropout))

	def forward(self, node_feat, adjacency_matrix, batch_graph):
		# Graph Convolution Layers Forward
		lv = 0
		output_matrix = node_feat
		cat_output_matrix = []
		while lv < len(self.latent_dim):
			output_matrix = self.conv_layers[lv](output_matrix, adjacency_matrix)
			cat_output_matrix.append(output_matrix)
			lv += 1

		if self.concat_tensors:
			return torch.cat(cat_output_matrix, 1)
		else:
			return output_matrix