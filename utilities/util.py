from __future__ import print_function
import numpy as np
import random
import torch
import json
import networkx as nx
from utilities.GNNGraph import GNNGraph
import matplotlib.pyplot as plt
import matplotlib

from utilities.lib.gnn_lib import GNNLIB


def graph_to_tensor(batch_graphs, node_feat_dim, edge_feat_dim, gpu):
	'''
	Converts networkx into tensors
	:param batch_graphs: List of GNNGraphs in the batch
	:param node_feat_dim: dimension of the node features
	:param edge_feat_dim: dimension of edge features
	:param gpu: defines whether CPU or GPU is used; 0 for CPU 1 for GPU
	:type batch_graphs: list of GNNGraphs
	:type node_feat_dim: integer
	:type edge_feat_dim: integer
	:param gpu: boolean
	:return node_feat: Sparse matrix of one-hot encoding of node features
	:return n2n_sp: Sparse matrix containing edge pairs of the graph.
	Indices consist of a tensor with two rows of node labels, where the first row refers to the
	node that the edge is coming out of and the second row refers to the nodes that the edge is going into
	Each column corresponds to an edge connecting two nodes.
	Values refer to the edge weights for each edge
	:return subg_sp: <Unsure> Number of nodes in the graph
	:rtype node_feat: Tensor
	:rtype n2n_sp: Tensor
	:rtype subg_sp: integer


	'''
	# Initialise counter for total number of nodes across all graphs in batch
	n_nodes = 0

	# Sanity checks: determine if the dataset have the following elements
	# Check if graph has node labels
	if batch_graphs[0].node_labels is not None:
		node_label_flag = True
		concat_label = []
	else:
		node_label_flag = False
		concat_label = None

	# Check if graph has node features
	if batch_graphs[0].node_features is not None:
		node_feat_flag = True
		concat_feat = []
	else:
		node_feat_flag = False
		concat_feat = None

	# Check if there are edge features
	if edge_feat_dim > 0:
		edge_feat_flag = True
		concat_edge_feat = []
	else:
		edge_feat_flag = False
		concat_edge_feat = None

	# Extraction loop: for each graph in batch, dissect graph elements and append to their respective lists
	for i in range(len(batch_graphs)):
		# Increment n_nodes by number of nodes in graph
		n_nodes += batch_graphs[i].number_of_nodes

		# Append graph node_labels (list) to concat_labels
		if node_label_flag == True:
			concat_label += batch_graphs[i].node_labels

		# Append graph node features (Tensor) to concat_feat
		if node_feat_flag == True:
			tmp = torch.from_numpy(batch_graphs[i].node_features).type(
				'torch.FloatTensor')
			concat_feat.append(tmp)

		# Append graph edge features (Tensor) into concat_edge_feat
		if edge_feat_flag == True:
			if batch_graphs[i].edge_features is not None:  # in case no edge in graph[i]
				tmp = torch.from_numpy(batch_graphs[i].edge_features).type(
					'torch.FloatTensor')
				concat_edge_feat.append(tmp)

	# Processing labels
	# Process node labels into one-hot embedding
	if node_label_flag == True:
		# Check if there are nodes with None as label (i.e. no node label)
		get_nodes_with_no_label = [index for index, value in enumerate(concat_label) if value is None]
		for index in get_nodes_with_no_label:
			concat_label[index] = 0

		# Perform one-hot embedding
		concat_label = torch.LongTensor(concat_label).view(-1, 1)
		node_label = torch.zeros(n_nodes, node_feat_dim)
		try:
			node_label.scatter_(1, concat_label, 1)
		except RuntimeError:
			# If there is unmapped node label
			batch_graphs[i].info()

		# Set all zeroes for nodes with no label
		for index in get_nodes_with_no_label:
			node_label[index] = torch.zeros(1, node_feat_dim)

	else:
		node_label = None

	# Process node features
	if node_feat_flag == True:
		node_feat = torch.cat(concat_feat, 0)
	else:
		node_feat = None

	# Concatenate one-hot embedding of node labels (node labels) with continuous node features
	if node_feat_flag and node_label_flag:
		node_feat = torch.cat([node_label.type_as(node_feat), node_feat], 1)
	elif node_feat_flag == False and node_label_flag == True:
		node_feat = node_label
	elif node_feat_flag == True and node_label_flag == False:
		pass
	else:
		# use all-one vector as node features
		node_feat = torch.ones(n_nodes, 1)

	# Process edge features
	if edge_feat_flag == True:
		edge_feat = torch.cat(concat_edge_feat, 0)
	else:
		edge_feat = None

	# Generate sparse matrices using library
	n2n_sp, e2n_sp, subg_sp = GNNLIB.PrepareSparseMatrices(batch_graphs)

	# If mode is GPU, enable cuda
	if torch.cuda.is_available() and gpu == 1:
		node_feat = node_feat.cuda()
		if edge_feat_flag == True and isinstance(node_feat, torch.cuda.FloatTensor):
			edge_feat = edge_feat.cuda()

	if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
		n2n_sp = n2n_sp.cuda()
		e2n_sp = e2n_sp.cuda()
		subg_sp = subg_sp.cuda()

	# If exists edge feature, concatenate to node feature vector
	if edge_feat is not None:
		input_edge_linear = edge_feat
		e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
		node_feat = torch.cat([node_feat, e2npool_input], 1)

	subg_sp = subg_sp.size()[0]

	return node_feat, n2n_sp, subg_sp


def hamming(s1, s2):
	"""Calculate the Hamming distance between two bit strings"""
	assert len(s1) == len(s2)
	return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def normalize_scores(scores, max_normalized_value):
	# Remember the signs
	signs = [x>= 0 for x in scores]

	# Normalise based on absolute value
	scores = [abs(score) for score in scores]

	minimum = min(scores)
	maximum = max(scores)

	scores = [(score - minimum) * max_normalized_value /
              (maximum - minimum) for score in scores]

	# Reapply the signs
	for i in range(len(scores)):
		if signs[i] is False:
			scores[i] *= -1

	return scores

def standardize_scores(scores):
	# Remember the signs
	signs = [x>= 0 for x in scores]

	# Standardize based on absolute value
	scores = [abs(score) for score in scores]
	maximum = max(scores)

	if maximum == 0:
		return scores

	scores = [score/maximum for score in scores]

	# Reapply the signs
	for i in range(len(scores)):
		if signs[i] is False:
			scores[i] *= -1

	return scores

def get_node_labels_dict(dataset):
	with open('data/%s/label_map.json' % dataset) as json_file:
		labels_dict = json.load(json_file)
		return labels_dict
