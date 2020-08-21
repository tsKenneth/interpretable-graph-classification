from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
	def __init__(self, hidden_size, num_class, output_dim, dropout=0.5,
		latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32], conv1d_kws=[0, 5],
		conv1d_activation='ReLU'):

		super(MLPClassifier, self).__init__()

		self.latent_dim = latent_dim
		self.k = k
		self.total_latent_dim = sum(latent_dim)
		conv1d_kws[0] = self.total_latent_dim

		self.output_dim = output_dim
		if output_dim == 0:
			dense_dim = int((k - 2) / 2 + 1)
			self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
			self.input_size = self.dense_dim
		else:
			self.input_size = latent_dim

		self.h1_weights = nn.Linear(self.input_size, hidden_size)
		self.h2_weights = nn.Linear(hidden_size, num_class)

		self.dropout = dropout
		if self.dropout > 0.001:
			self.dropout_layer = nn.Dropout(p=dropout)
		else:
			self.dropout_layer = None

		self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
		self.maxpool1d = nn.MaxPool1d(2, 2)
		self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

		if output_dim > 0:
			self.out_params = nn.Linear(self.dense_dim, output_dim)

		self.conv1d_activation_1 = eval('nn.{}()'.format(conv1d_activation))
		self.conv1d_activation_2 = eval('nn.{}()'.format(conv1d_activation))

	def forward(self, batch_graphs, graph_sizes):
		''' traditional 1d convlution and dense layers '''

		to_conv1d = batch_graphs.view((-1, 1, self.k * self.total_latent_dim))
		conv1d_res = self.conv1d_params1(to_conv1d)
		conv1d_res = self.conv1d_activation_1(conv1d_res)
		conv1d_res = self.maxpool1d(conv1d_res)
		conv1d_res = self.conv1d_params2(conv1d_res)
		conv1d_res = self.conv1d_activation_2(conv1d_res)

		to_dense = conv1d_res.view(len(graph_sizes), -1)

		if self.output_dim > 0:
			out_linear = self.out_params(to_dense)
			reluact_fp = self.conv1d_activation(out_linear)
		else:
			reluact_fp = to_dense
	
		h1 = self.h1_weights(reluact_fp)
		h1 = torch.relu(h1)

		if self.dropout_layer is not None:
			h1 = self.dropout_layer(h1)

		return self.h2_weights(h1)
