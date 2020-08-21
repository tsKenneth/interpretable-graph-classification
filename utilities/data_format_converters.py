import networkx as nx
import argparse
import os
import pickle
import pprint
import torch
import random

def callgraph_to_networkx(goodware_file_directory, malware_file_directory):
	list_of_goodware_graphs = pickle.load(open(goodware_file_directory, "rb"))
	list_of_malware_graphs = pickle.load(open(malware_file_directory, "rb"))

	node_feature_mapping_file = {
		'mov': 0,
		'call': 1,
		'lea': 2,
		'jmp': 3,
		'push': 4,
		'add': 5,
		'xor': 6,
		'cmp': 7,
		'int3': 8,
		'nop': 9,
		'pushl': 10,
		'dec': 11,
		'sub': 12,
		'insl': 13,
		'inc': 14,
		'jz': 15,
		'jnz': 16,
		'je': 17,
		'jne': 18,
		'ja': 19,
		'jna': 20,
		'js': 21,
		'jns': 22,
		'jl': 23,
		'jnl': 24,
		'jg': 25,
		'jng': 26}

	def iterate_graphs(unparsed_graph_list, graph_label):
		nxgraph_list = []
		for graph in unparsed_graph_list:
			node_attributes_dict = graph[0]
			edge_dict = graph[1]

			# Form basic graph structure using adjacency list with nodes and edges
			adjacency_list = []
			for node_from, node_to_list in edge_dict.items():
				if len(node_to_list) == 0:
					continue

				for node_to in node_to_list:
					adjacency_list.append((node_from, node_to))

			nxgraph = nx.from_edgelist(adjacency_list)

			# Add node features to graph. Convert node attribute to a feature vector by utilising count of each function
			for node in nxgraph.nodes():
				node_attributes = node_attributes_dict[node]

				node_attribute_list = [0 for _ in range(len(node_feature_mapping_file))]
				for node_feature, count in node_attributes.items():
					node_attribute_list[node_feature_mapping_file[node_feature]] += count

				nxgraph.nodes[node]['attribute'] = node_attribute_list

			# Add graph label
			nxgraph.graph['label'] = graph_label

			nxgraph_list.append(nxgraph)

		return nxgraph_list

	return iterate_graphs(list_of_goodware_graphs, 0) + iterate_graphs(list_of_malware_graphs, 1)

def subset_dataset(nxgraph_list, distribution_list):
	distribution_list = distribution_list.split(",")

	# Get labels of dataset
	graph_label_list = []
	for nxgraph in nxgraph_list:
		graph_label_list.append(nxgraph.graph["label"])

	graph_label_set = sorted(set(graph_label_list))

	# Check if number of elements in distribution list correspond with the number of graph labels found
	if len(graph_label_set) != len(distribution_list):
		print("Unable to create data subset, distribution list supplied does "
			  "not match number of classes found in dataset")
		return None

	output_nxgraph_list = [[] for _ in range(len(graph_label_set))]

	for nxgraph in nxgraph_list:
		output_nxgraph_list[graph_label_set.index(nxgraph.graph["label"])].append(nxgraph)

	for i in range(len(output_nxgraph_list)):
		random.shuffle(output_nxgraph_list[i])
		output_nxgraph_list[i] = output_nxgraph_list[i][:int(distribution_list[i])]

	return [nxgraph for nxgraph_sublist in output_nxgraph_list for nxgraph in nxgraph_sublist]



def dortmund_to_networkx(folder_directory, dataset_name):

	node_to_graph_mapping = {}
	# Process graph indicators, which maps node id to graph id
	with open("%s/%s_graph_indicator.txt" % (folder_directory, dataset_name), 'r') as file_graph_indicator:
		i = 1
		for line in file_graph_indicator:
			graph_indicator = int(line.strip("\n")) # Get rid of EOL character
			node_to_graph_mapping[i] = graph_indicator
			i += 1

	# Process graph labels, which maps graph id to graph label
	graph_label_mapping = []
	graph_label_mapping_dict = {}
	with open("%s/%s_graph_labels.txt" % (folder_directory, dataset_name), 'r') as file_graph_labels:
		for line in file_graph_labels:
			graph_label = int(line.strip("\n")) # Get rid of EOL character
			graph_label_mapping.append(graph_label)

	# Map graph_label depending on their values
	graph_label_set = set(graph_label_mapping)
	i = 0
	for graph_label in sorted(graph_label_set):
		graph_label_mapping_dict[graph_label] = i
		i += 1

	for i in range(len(graph_label_mapping)):
		graph_label_mapping[i] = graph_label_mapping_dict[graph_label_mapping[i]]

	# Process node labels, which maps node id to node label
	node_label_mapping = []
	try:
		with open("%s/%s_node_labels.txt" % (folder_directory, dataset_name), 'r') as file_node_labels:
			for line in file_node_labels:
				node_label = str(line.strip("\n"))  # Get rid of EOL character
				node_label_mapping.append(node_label)
	except IOError:
		print("dortmund_to_networkx: No node labels found for dataset %s" % dataset_name)

	# Process node features, which maps node id to node features
	node_attributes = []
	try:
		with open("%s/%s_node_attributes.txt" % (folder_directory, dataset_name), 'r') as file_node_features:
			for line in file_node_features:
				line = line.strip("\s\n")
				attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
				node_attributes.append(np.array(attrs))
	except IOError:
		print("dortmund_to_networkx: No node attributes found for dataset %s" % dataset_name)

	# Process adjacency matrix which specify edges between nodes
	adjacency_list = {i: [] for i in range(1, len(graph_label_mapping) + 1)}
	index_graph = {i: [] for i in range(1, len(graph_label_mapping) + 1)}

	with open("%s/%s_A.txt" % (folder_directory, dataset_name), 'r') as data_adjacency_list:
		for line in data_adjacency_list:
			line = line.strip("\n").split(",") # Get rid of EOL character and split line by comma
			n1, n2 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
			adjacency_list[node_to_graph_mapping[n1]].append((n1, n2))
			index_graph[node_to_graph_mapping[n1]] += [n1, n2]

	nxgraph_list = []
	for i in range(1, len(adjacency_list) + 1):
		# We ignore graphs with no edges
		if len(adjacency_list[i]) == 0:
			continue

		# Form basic graph structure using adjacency list with nodes and edges
		nxgraph = nx.from_edgelist(adjacency_list[i])

		# Add Graph label
		graph_label = graph_label_mapping[i - 1]
		nxgraph.graph['label'] = graph_label

		# Add Node labels
		for node in nxgraph.nodes():
			node_label = node_label_mapping[node-1]
			nxgraph.nodes[node]['label'] = node_label

		# Add Node attributes:
		if len(node_attributes) > 0:
			for node in nxgraph.nodes():
				nxgraph.nodes[node]['attribute'] = node_attributes[node - 1]

		nxgraph_list.append(nxgraph)

	return nxgraph_list


def graphsig_to_networkx(sample_path):
	graph_list = []
	with open(sample_path, 'r') as f:
		filedata = f.read().splitlines()
		i = 0
		while(True):
			if filedata[i] != '' and filedata[i][0] == '#':
				G = nx.Graph(label=0)
				i += 1
				no_of_nodes = int(filedata[i])
				for j in range(no_of_nodes):
					i += 1
					G.add_node(j, label=filedata[i])
				i += 1
				no_of_edges = int(filedata[i])
				for j in range(no_of_edges):
					i += 1
					edge_data = filedata[i].split()
					G.add_edge(int(edge_data[0]), int(edge_data[1]))
				graph_list.append(G)
			else:
				i += 1
			if i >= len(filedata):
				break
	return graph_list


if __name__ == '__main__':
	cmd_opt = argparse.ArgumentParser(description='Argparser for graph data formatter')
	cmd_opt.add_argument('-format', default='dortmund', help='Format to convert from')
	cmd_opt.add_argument('-path', help='Path to dataset, can be file or folder.')
	cmd_opt.add_argument('-outpath', default='default', help='Path to output the serialized pickle file,'
															 ' default is the same as input path. Must be specified '
															 'when format is adhoc')
	cmd_opt.add_argument('-distribution', help='Define the distribution of the subset, '
																  'separated by comma and ordered according to the'
																  ' value of graph labels')

	cmd_opt.add_argument('-adhocfunc', help='Specify which adhoc data formatting function to use')
	cmd_opt.add_argument('-adhocparam', help='Specify what adhoc parameters to be passed to the adhoc function,'
											  ' seperated by comma.')
	cmd_args, _ = cmd_opt.parse_known_args()

	random.seed(1800)

	# If conversion is adhoc, ignore the standard procedure
	if cmd_args.format == "adhoc":
		param_list = []
		for param in cmd_args.adhocparam.split(','):
			param_list.append("'" + param + "'")

		exec_string = "nxgraph_list = %s(%s)" % (str(cmd_args.adhocfunc), ','.join(param_list))
		exec(exec_string)
	else:
		path_is_file = os.path.isfile(cmd_args.path)

		# Get dataset name
		if path_is_file is True:
			dataset_name = cmd_args.path.split("/")[-2]
		else:
			dataset_name = cmd_args.path.split("/")[-1]

		# Begin conversion of format
		if cmd_args.format == "dortmund":
			if path_is_file is True:
				print("Error, dortmund format requires a folder")
				exit()

			folder_directory = cmd_args.path
			nxgraph_list = dortmund_to_networkx(folder_directory, dataset_name)
		elif cmd_args.format == "smiles":
			if path_is_file is False or cmd_args.path.split(".")[-1] != "csv":
				print("Error, smiles format requires a .csv file")
		else:
			print("Invalid or no dataset format specified!")
			exit()

	if cmd_args.distribution is not None:
		nxgraph_list = subset_dataset(nxgraph_list, cmd_args.distribution)

	# Check validity of output path
	if cmd_args.outpath == "default":
		if cmd_args.format == "adhoc":
			print("Output path(-outpath) needs to be specified for adhoc formatting")
		elif path_is_file is True:
			folder_dir = cmd_args.path.split("/")[:-1].join("/")
		else:
			folder_dir = cmd_args.path

		outpath = "%s/%s.p" % (folder_dir, dataset_name)
	else:
		outpath = cmd_args.outpath

	# Serialize the resulting networkx graph list using pickle
	with open(outpath, 'wb') as output_file:
		pickle.dump(nxgraph_list, output_file)
		print("Dataset formatted")
		output_file.close()
