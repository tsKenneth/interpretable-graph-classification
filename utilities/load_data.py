import numpy as np
import networkx as nx
import pickle
from sklearn.model_selection import StratifiedKFold

from utilities.GNNGraph import GNNGraph

def unserialize_pickle_file(path):
	'''
	:param path: filepath to the pickled dataset file
	:return: a list of GNNgraphs extracted from dataset
	'''
	# Unserialize the pickled file
	with open(path, 'rb') as pickled_file:
		nxgraph_list = pickle.load(pickled_file)
		pickled_file.close()

	# Begin converting to format compatible with this codebase
	graph_list = []
	graph_labels_mapping_dict = {}
	node_labels_mapping_dict = {}

	# Relabel nodes
	for i in range(len(nxgraph_list)):
		relabel_mapping = {}
		j = 0
		for n in nxgraph_list[i].nodes:
			relabel_mapping[n] = j
			j += 1

		nxgraph = nx.relabel_nodes(nxgraph_list[i], relabel_mapping, copy=True)

		nxgraph_list[i] = nxgraph

	# Prepare graph mapping dict and node mapping dict to retain the order of the graph and node labels
	graph_mapping_list = []
	node_mapping_list = []
	for nxgraph in nxgraph_list:
		graph_mapping_list.append(nxgraph.graph['label'])

	graph_label_set = set(graph_mapping_list)
	i = 0
	for graph_label in sorted(graph_label_set):
		graph_labels_mapping_dict[str(graph_label)] = i
		i += 1

	if "label" in nxgraph.nodes[0].keys():
		for nxgraph in nxgraph_list:
			for node in nxgraph.nodes:
				node_mapping_list.append(nxgraph.nodes[node]['label'])

		node_label_set = set(node_mapping_list)
		j = 0

		for node_label in sorted(node_label_set):
			node_labels_mapping_dict[str(node_label)] = j
			j += 1

		# Add an additional node label for possible use in occlusion later in metrics.py
		new_node_label = len(node_labels_mapping_dict)
		node_labels_mapping_dict["UNKNOWN"] = new_node_label

	else:
		node_labels_mapping_dict = {}


	# Extract graph labels
	graph_id = 0
	for nxgraph in nxgraph_list:
		# Get graph label
		graph_label = graph_labels_mapping_dict[str(nxgraph.graph['label'])]

		# Get node labels
		node_labels = []

		if "label" in nxgraph.nodes[0].keys():
			for node in nxgraph.nodes:
				node_label = str(nxgraph.nodes[node]['label'])
				node_labels.append(node_labels_mapping_dict[node_label])
			node_label_flag = True
		else:
			node_label_flag = False
			node_labels = None

		# Get node features/attributes
		node_features = []
		if "attribute" in nxgraph.nodes[0].keys():
			for node in nxgraph.nodes:
				node_features.append(np.array(nxgraph.nodes[node]['attribute']))

			node_features = np.stack(node_features, axis=0)
			node_feature_flag = True
		else:
			node_feature_flag = False
			node_features = None

		graph_list.append(GNNGraph(graph_id, nxgraph, graph_label, node_labels, node_features))
		graph_id += 1

	return graph_list


def unserialize_pickle(dataset_name):
	# Unserialize the pickled file
	with open("data/%s/%s.p" % (dataset_name, dataset_name), 'rb') as pickled_file:
		nxgraph_list = pickle.load(pickled_file)
		pickled_file.close()

	# Begin converting to format compatible with this codebase
	graph_list = []
	graph_labels_mapping_dict = {}
	node_labels_mapping_dict = {}

	# Relabel nodes
	for i in range(len(nxgraph_list)):
		relabel_mapping = {}
		j = 0
		for n in nxgraph_list[i].nodes:
			relabel_mapping[n] = j
			j += 1

		nxgraph = nx.relabel_nodes(nxgraph_list[i], relabel_mapping, copy=True)

		nxgraph_list[i] = nxgraph

	# Prepare graph mapping dict and node mapping dict to retain the order of the graph and node labels
	graph_mapping_list = []
	node_mapping_list = []
	for nxgraph in nxgraph_list:
		graph_mapping_list.append(nxgraph.graph['label'])

	graph_label_set = set(graph_mapping_list)
	i = 0
	for graph_label in sorted(graph_label_set):
		graph_labels_mapping_dict[str(graph_label)] = i
		i += 1

	if "label" in nxgraph.nodes[0].keys():
		for nxgraph in nxgraph_list:
			for node in nxgraph.nodes:
				node_mapping_list.append(nxgraph.nodes[node]['label'])

		node_label_set = set(node_mapping_list)

		j = 0
		for node_label in sorted(node_label_set):
			node_labels_mapping_dict[str(node_label)] = j
			j += 1

		# Add an additional node label for possible use in occlusion later in metrics.py
		new_node_label = len(node_labels_mapping_dict)
		node_labels_mapping_dict["UNKNOWN"] = new_node_label

	else:
		node_labels_mapping_dict = {}

	# Extract graph labels
	graph_id = 0
	for nxgraph in nxgraph_list:
		# Get graph label
		graph_label = graph_labels_mapping_dict[str(nxgraph.graph['label'])]

		# Get node labels
		node_labels = []

		if "label" in nxgraph.nodes[0].keys():
			for node in nxgraph.nodes:
				node_label = str(nxgraph.nodes[node]['label'])
				node_labels.append(node_labels_mapping_dict[node_label])
			node_label_flag = True
		else:
			node_label_flag = False
			node_labels = None

		# Get node features/attributes
		node_features = []
		if "attribute" in nxgraph.nodes[0].keys():
			for node in nxgraph.nodes:
				node_features.append(np.array(nxgraph.nodes[node]['attribute']))

			node_features = np.stack(node_features, axis=0)
			node_feature_flag = True
		else:
			node_feature_flag = False
			node_features = None

		graph_list.append(GNNGraph(graph_id, nxgraph, graph_label, node_labels, node_features))
		graph_id += 1

	return graph_list, graph_labels_mapping_dict, node_labels_mapping_dict, node_label_flag, node_feature_flag

# load_data(): Loads pickled dataset
def load_model_data(dataset_name, k_fold=5, dataset_autobalance=False, print_dataset_info=True):
	'''

	:param dataset_name: name of the dataset to use
	:param k_fold: the number of folds to split the dataset
	:param test_number: if specified, use this to split dataset instead of k_fold
	:param dataset_autobalance: whether to balances dataset by class distribtuion if it is too skewed.
	:param print_dataset_info: whether to print information on the dataset
	:return:
	'''
	print('load_data.py load_model_data(): Unserialising pickled dataset into Graph objects')

	# Perform unserialisation
	graph_list, graph_labels_mapping_dict, node_labels_mapping_dict, node_label_flag, node_feature_flag =\
		unserialize_pickle(dataset_name)

	# Count the number of labels, and form a graph label list for kfold split later
	label_count_list = [0 for _ in range(len(graph_labels_mapping_dict))]
	graph_labels = []
	for graph in graph_list:
		label_count_list[graph.label] += 1
		graph_labels.append(graph.label)

	# If the dataset is too imbalanced, perform balancing operation using under-sampling
	if dataset_autobalance and len(label_count_list) == 2:
		balance_ratio = min(label_count_list[0] / label_count_list[1], label_count_list[1] / label_count_list[0])
		ideal_balance_ratio = 0.5

		if balance_ratio < ideal_balance_ratio:
			print("load_data.py: Dataset is too imbalanced at %s, restoring to atleast %s now." %
				  (str(round(balance_ratio, 3)), str(ideal_balance_ratio)))
			if label_count_list[0] > label_count_list[1]:
				endslice = round(len(graph_split[1]) / ideal_balance_ratio - len(graph_split[1]))
				graph_list = graph_split[0][:endslice] + graph_split[1]
				graph_labels = [1 for _ in range(endslice)] + [0 for _ in range(len(graph_split[1]))]
			else:
				endslice = round(len(graph_split[0]) / ideal_balance_ratio - len(graph_split[0]))
				graph_list = graph_split[1][:endslice] + graph_split[0]
				graph_labels = [1 for _ in range(endslice)] + [0 for _ in range(len(graph_split[0]))]

		# Recalculate label_count_list again:
		label_count_list = [0 for _ in len(label_count_list)]
		for label in graph_labels:
			label_count_list[label] += 1

	# Set useful dataset features into a dictionary to be passed to main later
	dataset_features = {}
	dataset_features['name'] = dataset_name
	dataset_features['num_class'] = len(graph_labels_mapping_dict)
	dataset_features['label_dict'] = graph_labels_mapping_dict
	dataset_features['have_node_labels'] = node_label_flag
	dataset_features['have_node_attributions'] = node_feature_flag
	dataset_features['node_dict'] = node_labels_mapping_dict
	dataset_features['feat_dim'] = len(node_labels_mapping_dict)
	dataset_features['edge_feat_dim'] = 0
	graph_sizes_list = [graph.number_of_nodes for graph in graph_list]
	dataset_features['max_num_nodes'] = max(graph_sizes_list)
	dataset_features['avg_num_nodes'] = round(sum(graph_sizes_list)/len(graph_sizes_list))
	dataset_features['graph_sizes_list'] = graph_sizes_list

	if node_feature_flag == True:
		dataset_features['attr_dim'] = graph_list[0].node_features.shape[1]
	else:
		dataset_features['attr_dim'] = 0

	# If verbose on dataset features
	if print_dataset_info:
		# Get class distribution of graphs
		class_distribution_dict = {}
		inverse_graph_label_dict = {v: k for k, v in graph_labels_mapping_dict.items()}
		inverse_node_label_dict = {v: k for k, v in node_labels_mapping_dict.items()}

		for i in range(len(label_count_list)):
			class_distribution_dict[inverse_graph_label_dict[i]] = label_count_list[i]

		# Get node statistics
		unique_node_labels_count_list = []
		unique_node_features_per_graph_count_list = []
		unique_node_features_per_node_count_list = []
		node_labels_count_dict = {}

		if graph.node_labels is not None:
			for graph in graph_list:
				unique_node_labels_count_list.append(len(graph.unique_node_labels))
				for node_label in graph.node_labels:
					original_node_label = inverse_node_label_dict[node_label]
					if original_node_label not in node_labels_count_dict.keys():
						node_labels_count_dict[original_node_label] = 1
					else:
						node_labels_count_dict[original_node_label] += 1

		if graph.node_features is not None:
			for graph in graph_list:
				sum_node_features_in_graph = [sum(x) for x in zip(*graph.node_features)]
				unique_node_features_per_graph_count_list.append(sum([_ > 0 for _ in sum_node_features_in_graph]))
				for node_feature in graph.node_features:
					unique_node_features_per_node_count_list.append(sum([_ > 0 for _ in node_feature]))

		# Get Edge statistics
		edge_count_list = []
		for graph in graph_list:
			edge_count_list.append(len(graph.edge_pairs)/2)

		# Build verbose message
		dataset_features_string = "==== Dataset Information ====\n"
		dataset_features_string += "== General Information == \n"
		dataset_features_string += "Number of graphs: " + str(len(graph_list)) + "\n"
		dataset_features_string += "Number of classes: " + str(dataset_features['num_class']) + "\n"
		dataset_features_string += "Class distribution: \n"

		for key in sorted(class_distribution_dict.keys()):
			dataset_features_string += '{}:{} '.format(key, class_distribution_dict[key])

		dataset_features_string += "\n\n"
		dataset_features_string += "== Node information== \n"
		dataset_features_string += "Average number of nodes: " + str(dataset_features['avg_num_nodes']) + "\n"
		dataset_features_string += "Average number of edges (undirected): " + \
								   str(round(sum(edge_count_list)/len(graph_list))) + "\n"
		dataset_features_string += "Max number of nodes: " + str(dataset_features['max_num_nodes']) + "\n"

		if graph.node_labels is not None:
			dataset_features_string += "Number of distinct node labels: " + str(len(node_labels_count_dict)) + "\n"
			dataset_features_string += "Average number of distinct node labels: " + \
									   str(round(sum(unique_node_labels_count_list)/len(graph_list))) + "\n"
			dataset_features_string += "Node labels distribution: " + "\n"

			for node_label in sorted(node_labels_count_dict.keys()):
				dataset_features_string += '{}:{} '.format(node_label, node_labels_count_dict[node_label])

		if graph.node_features is not None:
			dataset_features_string += "Average number of distinct node features per graph: " + \
								   str(round(sum(unique_node_features_per_graph_count_list)/len(graph_list))) + "\n"

			dataset_features_string += "Average number of distinct node features per node: " + \
								   str(round(sum(unique_node_features_per_node_count_list)/
											 len(unique_node_features_per_node_count_list))) + "\n"
		dataset_features_string += "\n"

		dataset_features["dataset_info"] = dataset_features_string
		print(dataset_features_string)

	# If no test number is specified, use stratified KFold sampling for train test split
	stratified_KFold = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=None)
	i = 0

	train_graphs = []
	test_graphs = []
	for train_index, test_index in stratified_KFold.split(graph_list, graph_labels):
		train_graphs.append([graph_list[i] for i in train_index])
		test_graphs.append([graph_list[i] for i in test_index])

	return train_graphs, test_graphs, dataset_features
