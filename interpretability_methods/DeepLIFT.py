import torch
import json
import random
import networkx as nx

from time import perf_counter
from os import path
from copy import deepcopy
from captum.attr import DeepLift
from utilities.util import graph_to_tensor, standardize_scores

def get_isomorphic_pairs(dataset_name, graph_list, k_fold, current_fold=None, max_pairs=5):
	'''
		Get isomorphic pairs to serve as a baseline to be used in DeepLIFT
	:param data_file_name: name of the dataset
	:param graph_list: a list of graphs to obtain isomorphic pairs
	:param k_fold: the number of k folds used in this dataset, to account for isomorphic index changing every fold
	:param current_fold: the fold where DeepLIFT is being currently applied to
	:param max_pairs: max number of pairs to find. Set this to be low to decrease execution time
	:return: two sets of list that contain indices of the isomorphic graphs
	'''

	# Check if temporary file storing isomorphic pairs exist.
	# This is used to reduce time for repeated experiments
	indexes_by_fold = {}
	if path.exists("tmp/deeplift/isopairs_%s_folds_%s.json" % (dataset_name, k_fold)):
		with open("tmp/deeplift/isopairs_%s_folds_%s.json" % (dataset_name, k_fold), 'r') as f:
			indexes_by_fold = json.load(f)
			f.close()

		if str(current_fold) in indexes_by_fold.keys():
			class_0_indices = indexes_by_fold[str(current_fold)][0]
			class_1_indices = indexes_by_fold[str(current_fold)][1]

			# Check if existing isomorphic pairs found is greater than the max pairs needed
			if len(class_0_indices) == 0:
				# If no isomorphic pairs stored, return None
				return None, None
			elif len(class_0_indices) <= max_pairs:
				# If exact or less than required, return as is
				return [graph_list[i] for i in class_0_indices], [graph_list[i] for i in class_1_indices]
			elif len(class_0_indices) > max_pairs:
				# If more pairs than required, slice
				return [graph_list[i] for i in class_0_indices[:max_pairs-1]],\
					   [graph_list[i] for i in class_1_indices[:max_pairs-1]]

	# If no such file exist, then run loop to find isomorphic pairs
	# Split input graph set by class
	i = 0
	iso_graph_indices = [[], []]
	for GNNgraph in graph_list:
		iso_graph_indices[GNNgraph.label].append(i)
		i += 1

	# Function currently only supports binary class labels
	if len(iso_graph_indices) != 2:
		print("ERROR: Only binary graph labels are supported for obtaining isomorphic pairs")
		exit()

	# Run loop to find isomorphic graphs. Exit when max pairs are found
	print("Finding isomorphic graphs. This may take awhile.")
	class_0_indices = []
	class_1_indices = []
	pairs_found = 0
	max_pairs_reached = False
	for GNNgraph_0_index in iso_graph_indices[0]:
		if max_pairs_reached:
			break

		for GNNgraph_1_index in iso_graph_indices[1]:
			if nx.is_isomorphic(graph_list[GNNgraph_0_index].to_nxgraph(),
				graph_list[GNNgraph_1_index].to_nxgraph()):
				class_0_indices.append(GNNgraph_0_index)
				class_1_indices.append(GNNgraph_1_index)
				pairs_found += 1

				if pairs_found >= max_pairs:
					max_pairs_reached = True
					break

	if pairs_found > 0:
		indexes_by_fold[str(current_fold)] = [class_0_indices, class_1_indices]

		with open("tmp/deeplift/isopairs_%s_folds_%s.json" % (dataset_name, k_fold), 'w') as f:
			f.write(json.dumps(indexes_by_fold))
			f.close()

	# Return the graphs based on the index of the isomorphic pairs
	return [graph_list[i] for i in class_0_indices], [graph_list[i] for i in class_1_indices]

def DeepLIFT(classifier_model, config, dataset_features, GNNgraph_list, current_fold, cuda=0):
	'''
		:param classifier_model: trained classifier model
		:param config: parsed configuration file of config.yml
		:param dataset_features: a dictionary of dataset features obtained from load_data.py
		:param GNNgraph_list: a list of GNNgraphs obtained from the dataset
		:param cuda: whether to use GPU to perform conversion to tensor
	'''
	# Initialise settings
	config = config
	interpretability_config = config["interpretability_methods"]["DeepLIFT"]
	dataset_features = dataset_features

	# Perform deeplift on the classifier model
	dl = DeepLift(classifier_model)

	output_for_metrics_calculation = []
	output_for_generating_saliency_map = {}

	# Obtain attribution score for use in qualitative metrics
	tmp_timing_list = []

	for GNNgraph in GNNgraph_list:
		output = {'graph': GNNgraph}
		for _, label in dataset_features["label_dict"].items():
			# Relabel all just in case, may only relabel those that need relabelling
			# if performance is poor
			original_label = GNNgraph.label
			GNNgraph.label = label

			node_feat, n2n, subg = graph_to_tensor(
				[GNNgraph], dataset_features["feat_dim"],
				dataset_features["edge_feat_dim"], cuda)

			start_generation = perf_counter()
			attribution = dl.attribute(node_feat,
								   additional_forward_args=(n2n, subg, [GNNgraph]),
								   target=label)
			tmp_timing_list.append(perf_counter() - start_generation)
			attribution_score = torch.sum(attribution, dim=1).tolist()
			attribution_score = standardize_scores(attribution_score)

			GNNgraph.label = original_label

			output[label] = attribution_score
		output_for_metrics_calculation.append(output)

	execution_time = sum(tmp_timing_list)/(len(tmp_timing_list))

	# Obtain attribution score for use in generating saliency map for comparison with zero tensors
	if interpretability_config["compare_with_zero_tensor"] is True:
		if interpretability_config["sample_ids"] is not None:
			if ',' in str(interpretability_config["sample_ids"]):
				sample_graph_id_list = list(map(int, interpretability_config["sample_ids"].split(',')))
			else:
				sample_graph_id_list = [int(interpretability_config["sample_ids"])]

			output_for_generating_saliency_map.update({"layergradcam_%s_%s" % (str(assign_type), str(label)): []
													   for _, label in dataset_features["label_dict"].items()})

			for index in range(len(output_for_metrics_calculation)):
				tmp_output = output_for_metrics_calculation[index]
				tmp_label = tmp_output['graph'].label
				if tmp_output['graph'].graph_id in sample_graph_id_list:
					element_name = "layergradcam_%s_%s" % (str(assign_type), str(tmp_label))
					output_for_generating_saliency_map[element_name].append(
						(tmp_output['graph'], tmp_output[tmp_label]))

		elif interpretability_config["number_of_zero_tensor_samples"] > 0:
			# Randomly sample from existing list:
			graph_idxes = list(range(len(output_for_metrics_calculation)))
			random.shuffle(graph_idxes)
			output_for_generating_saliency_map.update({"deeplift_zero_tensor_class_%s" % str(label): []
													   for _, label in dataset_features["label_dict"].items()})

			# Begin appending found samples
			for index in graph_idxes:
				tmp_label = output_for_metrics_calculation[index]['graph'].label
				element_name = "deeplift_zero_tensor_class_%s" % str(tmp_label)
				if len(output_for_generating_saliency_map[element_name]) < interpretability_config["number_of_zero_tensor_samples"]:
					output_for_generating_saliency_map[element_name].append(
						(output_for_metrics_calculation[index]['graph'], output_for_metrics_calculation[index][tmp_label]))

	# Obtain attribution score for use in generating saliency map for comparison with isomers
	if interpretability_config["compare_with_isomorphic_samples"] is True:
		if dataset_features["num_class"] != 2:
			print("DeepLIFT.py: Comparing with isomorphic samples is only possible in binary classification tasks.")
		else:
			# Get all isomorphic pairs
			class_0_graphs, class_1_graphs = get_isomorphic_pairs(
				dataset_features["name"], GNNgraph_list, config["run"]["k_fold"], current_fold,
				interpretability_config["number_of_isomorphic_sample_pairs"])

			# Generate attribution scores for the isomorphic pairs
			if class_0_graphs == None:
				pass
			elif len(class_0_graphs) == 0 or len(class_1_graphs) == 0:
				print("DeepLIFT: No isomorphic pairs found for test dataset")
			else:
				output_for_generating_saliency_map["deeplift_isomorphic_class_0"] = []
				output_for_generating_saliency_map["deeplift_isomorphic_class_1"] = []

				for graph_0, graph_1 in zip(class_0_graphs, class_1_graphs):
					node_feat_0, n2n, subg = graph_to_tensor(
						[graph_0], dataset_features["feat_dim"],
						dataset_features["edge_feat_dim"], cuda)

					node_feat_1, _, _ = graph_to_tensor(
						[graph_1], dataset_features["feat_dim"],
						dataset_features["edge_feat_dim"], cuda)

					attribution_0 = dl.attribute(node_feat_0,
						additional_forward_args=(n2n, subg, [graph_0]),
						baselines=node_feat_1,
						target=graph_0.label)

					attribution_1 = dl.attribute(node_feat_1,
						additional_forward_args=(n2n, subg, [graph_1]),
						baselines=node_feat_0,
						target=graph_1.label)

					attribution_score_0 = torch.sum(attribution_0, dim=1).tolist()
					attribution_score_1 = torch.sum(attribution_1, dim=1).tolist()

					attribution_score_0 = standardize_scores(attribution_score_0)
					attribution_score_1 = standardize_scores(attribution_score_1)

					output_for_generating_saliency_map["deeplift_isomorphic_class_0"].append(
						(graph_0, attribution_score_0))
					output_for_generating_saliency_map["deeplift_isomorphic_class_1"].append(
						(graph_1, attribution_score_1))

	return output_for_metrics_calculation, output_for_generating_saliency_map, execution_time