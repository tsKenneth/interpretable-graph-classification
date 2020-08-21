import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.layers.lib.layer_util import gnn_spmm

class GraphConvolutionLayer_GCN(nn.Module):
	'''
		A single graph convolution layer, as defined by Kipf and Welling
	'''
	def __init__(self,
		 		input_dim,
				latent_dim,
				dropout=0.0):
		# Intialise settings
		super(GraphConvolutionLayer_GCN, self).__init__()
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.dropout = dropout

		self.conv_params = nn.Linear(input_dim, latent_dim)

		if self.dropout > 0.001:
			self.dropout_layer = nn.Dropout(p=dropout)
		else:
			self.dropout_layer = None

	def forward(self, input_tensor, adjacency_matrix, node_degree):
		# Graph Convolution Layer Forward
		dense_adjacency_matrix = adjacency_matrix.to_dense() # A
		dense_adjacency_matrix = dense_adjacency_matrix + torch.eye(list(dense_adjacency_matrix.size())[0]) # A~ = (A + I)
		normalised_degree = torch.rsqrt(node_degree) # D~^-(1/2)
		adjacency_matrix = normalised_degree.mul(dense_adjacency_matrix).mul(normalised_degree) # A~' D~^(-1/2) A~ D~^(-1/2)
		adjacency_matrix = adjacency_matrix.to_sparse()

		adjacency_matrixpool = gnn_spmm(adjacency_matrix, input_tensor) + input_tensor  # Y = A~' * X
		node_linear = self.conv_params(adjacency_matrixpool)  # Y = Y * W
		output_tensor = torch.relu(node_linear)
		if self.dropout_layer is not None:
			output_tensor = self.dropout_layer(output_tensor)
		return output_tensor
	
class GraphConvolutionLayer_DGCNN(nn.Module):
	'''
		A single graph convolution layer, using propagation matrix defined by Zhang et al. in DGCNN
	'''
	def __init__(self,
		 		input_dim,
				latent_dim,
				dropout=0.0):
		# Intialise settings
		super(GraphConvolutionLayer_DGCNN, self).__init__()
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.dropout = dropout

		if self.dropout > 0.001:
			self.dropout_layer = nn.Dropout(p=dropout)
		else:
			self.dropout_layer = None

		self.conv_params = nn.Linear(input_dim, latent_dim)

	def forward(self, input_tensor, adjacency_matrix, node_degree_matrix):
		# Graph Convolution Layer Forward
		adjacency_matrixpool = gnn_spmm(adjacency_matrix, input_tensor) + input_tensor  # Y = (A + I) * X
		node_linear = self.conv_params(adjacency_matrixpool)  # Y = Y * W
		normalized_linear = node_linear.div(node_degree_matrix)  # Y = D^-1 * Y
		output_tensor = torch.tanh(normalized_linear)
		if self.dropout_layer is not None:
			output_tensor = self.dropout_layer(output_tensor)
		return output_tensor

class GraphConvolutionLayer_DGCNN_Dense(nn.Module):
	'''
		A single graph convolution layer, using propagation matrix defined by Zhang et al. in DGCNN
		Accepts a dense matrix for adjacency matrix as opposed to a sparse matrix
	'''
	def __init__(self,
		 		input_dim,
				latent_dim,
				dropout=0.0):
		# Intialise settings
		super(GraphConvolutionLayer_DGCNN_Dense, self).__init__()
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.dropout = dropout

		if self.dropout > 0.001:
			self.dropout_layer = nn.Dropout(p=dropout)
		else:
			self.dropout_layer = None

		self.conv_params = nn.Linear(input_dim, latent_dim)

	def forward(self, input_tensor, adjacency_matrix, node_degree_matrix):
		# Graph Convolution Layer Forward
		adjacency_matrixpool = torch.mm(adjacency_matrix, input_tensor) + input_tensor  # Y = (A + I) * X
		node_linear = self.conv_params(adjacency_matrixpool)  # Y = Y * W
		normalized_linear = node_linear.div(node_degree_matrix)  # Y = D^-1 * Y
		output_tensor = torch.tanh(normalized_linear)
		if self.dropout_layer is not None:
			output_tensor = self.dropout_layer(output_tensor)
		return output_tensor

class GraphConvolutionLayer_GraphSAGE(nn.Module):
	'''
		A single graph convolution layer, as defined in GraphSAGE
	'''
	def __init__(self,
		 		input_dim,
				latent_dim,
				add_self=False,
				normalize_embedding=True,
				bias=True,
				dropout=0.0):
		# Intialise settings
		super(GraphConvolutionLayer_GraphSAGE, self).__init__()
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.dropout = dropout
		self.normalize_embedding = normalize_embedding
		self.add_self = add_self

		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(latent_dim))
		else:
			self.bias = None

		if self.dropout > 0.001:
			self.dropout_layer = nn.Dropout(p=dropout)
		else:
			self.dropout_layer = None

		self.weight = nn.Parameter(torch.FloatTensor(input_dim, latent_dim))


	def forward(self, input_tensor, adjacency_matrix):
		# Graph Convolution Layer Forward
		if self.dropout > 0.001:
			input_tensor = self.dropout_layer(input_tensor)

		adjacency_matrixpool = torch.mm(adjacency_matrix, input_tensor) + input_tensor  # Y = A * X
		if self.add_self:
			adjacency_matrixpool + input_tensor # Y = (A + I) * X

		node_linear = torch.matmul(adjacency_matrixpool, self.weight) # Y * W

		if self.bias is not None:
			node_linear = node_linear + self.bias

		if self.normalize_embedding is True:
			node_linear = F.normalize(node_linear, p=2, dim=1)

		output_tensor = torch.relu(node_linear)
		return output_tensor