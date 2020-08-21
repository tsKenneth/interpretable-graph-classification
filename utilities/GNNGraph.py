import networkx as nx
import numpy as np

class GNNGraph(object):
	def __init__(self, graph_id, nxgraph, label, node_labels=None, node_features=None):
		'''
			graph_id: position of the graph in the dataset
			nxgraph: a networkx graph
			label: an integer graph label that describes the class
			undirected: boolean to flag if GNN Graph is undirected
			node_labels: a list of integer node labels
			node_features: a numpy array of continuous node features
		'''
		self.graph_id = graph_id
		self.number_of_nodes = len(nxgraph.nodes())
		self.label = label
		self.node_features = node_features  # numpy array (node_num * feature_dim)
		self.node_degrees = list(dict(nxgraph.degree).values())

		if node_labels is not None:
			self.node_labels = node_labels
			self.unique_node_labels = set(node_labels)
		else:
			self.node_labels = []
			self.unique_node_labels = ()

		# Process edge pairs
		if len(nxgraph.edges()) != 0:
			x, y = zip(*nxgraph.edges())
			self.number_of_edges = len(x)
			self.edge_pairs = np.ndarray(shape=(self.number_of_edges, 2), dtype=np.int32)
			self.edge_pairs[:, 0] = x
			self.edge_pairs[:, 1] = y
			self.edge_pairs = self.edge_pairs.flatten()
		else:
			self.number_of_edges = 0
			self.edge_pairs = []

		# Process Edge Features
		self.edge_features = None
		if nx.get_edge_attributes(nxgraph, 'features'):
			# Make sure edges have an attribute 'features' (1 * feature_dim numpy array)
			edge_features = nx.get_edge_attributes(nxgraph, 'features')
			assert (type(edge_features.values()[0]) == np.ndarray)
			# Need to rearrange edge_features using the e2n edge order
			edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
			keys = sorted(edge_features)
			self.edge_features = []
			for edge in keys:
				self.edge_features.append(edge_features[edge])
				self.edge_features.append(edge_features[edge])  # add reversed edges
			self.edge_features = np.concatenate(self.edge_features, 0)

	def info(self, graph_labels_mapping_dict = None, node_labels_mapping_dict = None):
		info_string = ""

		# Convert graph labels to original label
		if graph_labels_mapping_dict is not None:
			inverse_graph_mapping_dict = {v: k for k, v in graph_labels_mapping_dict.items()}
			original_graph_label = inverse_graph_mapping_dict[self.label]
		else:
			print("WARNING: Graph label are in a re-mapped form. Supply the graph label mapping dictionary to convert.")
			original_graph_label = self.label

		# Convert node labels to original label
		if node_labels_mapping_dict is not None:
			original_node_labels = []
			inverse_node_mapping_dict = {v: k for k, v in node_labels_mapping_dict.items()}
			for node_label in self.node_labels:
				original_node_labels.append(inverse_node_mapping_dict[node_label])

		else:
			print("WARNING: Node labels are in a re-mapped form. Supply the node label mapping dictionary to convert.")
			original_node_labels = self.node_labels

		info_string += "Graph Label: %s \n  Number of nodes: %s \n" \
			"  Node Labels: %s \n  Node Degrees: %s \n  Total Degree: %s\n" \
			"  Number of Edges: %s \n  Edge Pairs: %s \n" % \
			(original_graph_label, self.number_of_nodes, original_node_labels, self.node_degrees,
			sum(self.node_degrees), self.number_of_edges, self.edge_pairs)

		print(info_string)

	def to_nxgraph(self):
		nxgraph = nx.Graph(label=self.label)
		for graph_node in range(self.number_of_nodes):
			if len(self.node_labels) > 0:
				nxgraph.add_node(graph_node, label=self.node_labels[graph_node])
			else:
				nxgraph.add_node(graph_node)

		for pair_from, pair_to in zip(self.edge_pairs[0::2], self.edge_pairs[1::2]):
			nxgraph.add_edge(pair_from, pair_to)

		return nxgraph

	# TODO Future implementation, remove reliance on gnnlib to prepare sparse matrix
	# def prepare_sparse_matrices(self, batch_graph):
	# 	for graph in batch_graph:
	# 		number_of_edges = graph.number_of_edges
	# 		total_num_nodes = graph.number_of_nodes
	#
	# 		print(graph.edge_pairs)
	#
	# 		# Node to Node
	# 		edge_pair_chunks = sorted(graph.edge_pairs, key=itemgetter(0, 1))
	#
	# 		edge_pair_to = []
	# 		edge_pair_from = []
	#
	# 		for pair in edge_pair_chunks:
	# 			edge_pair_from.append(pair[0])
	# 			edge_pair_to.append(pair[1])
	# 		n2n_idxes = torch.LongTensor([edge_pair_from, edge_pair_to])
	# 		n2n_vals = torch.ones(number_of_edges*2, dtype=torch.float64)
	#
	# 		# Edge to Edge
	#
	# 	print(n2n_vals)
	#
	# 	n2n_sp = torch.sparse.FloatTensor(n2n_idxes, n2n_vals, torch.Size([total_num_nodes, total_num_nodes]))
	# 	print(n2n_sp)
	#
	# 	exit()
	#
	# 	return n2n_sp, e2n_sp, subg_sp