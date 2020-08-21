import torch
import json
import random

from time import perf_counter
from os import path
from copy import deepcopy
from captum.attr import LayerGradCam
from utilities.util import graph_to_tensor, standardize_scores

def LayerGradCAM(classifier_model, config, dataset_features, GNNgraph_list, current_fold=None, cuda=0):
	'''
		Attribute to input layer using soft assign
		:param classifier_model: trained classifier model
		:param config: parsed configuration file of config.yml
		:param dataset_features: a dictionary of dataset features obtained from load_data.py
		:param GNNgraph_list: a list of GNNgraphs obtained from the dataset
		:param current_fold: has no use in this method
		:param cuda: whether to use GPU to perform conversion to tensor
	'''
	# Initialise settings
	config = config
	interpretability_config = config["interpretability_methods"]["LayerGradCAM"]
	dataset_features = dataset_features
	assign_type = interpretability_config["assign_attribution"]

	# Perform grad cam on the classifier model and on a specific layer
	layer_idx = interpretability_config["layer"]
	if layer_idx == 0:
		gc = LayerGradCam(classifier_model, classifier_model.graph_convolution)
	else:
		gc = LayerGradCam(classifier_model, classifier_model.conv_modules[layer_idx-1])

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

			attribution = gc.attribute(node_feat,
									   additional_forward_args=(n2n, subg, [GNNgraph]),
									   target=label, relu_attributions = True)

			# Attribute to the input layer using the assign method specified
			reverse_assign_tensor_list = []
			for i in range(1, layer_idx + 1):
				assign_tensor = classifier_model.cur_assign_tensor_list[i - 1]
				max_index = torch.argmax(assign_tensor, dim=1, keepdim=True)
				if assign_type == "hard":
					reverse_assign_tensor = torch.transpose(
						torch.zeros(assign_tensor.size()).scatter_(1, max_index, value=1), 0, 1)
				else:
					reverse_assign_tensor = torch.transpose(assign_tensor, 0, 1)

				reverse_assign_tensor_list.append(reverse_assign_tensor)

			attribution = torch.transpose(attribution, 0, 1)

			for reverse_tensor in reversed(reverse_assign_tensor_list):
				attribution = attribution @ reverse_tensor

			attribution = torch.transpose(attribution, 0, 1)
			tmp_timing_list.append(perf_counter() - start_generation)

			attribution_score = torch.sum(attribution, dim=1).tolist()
			attribution_score = standardize_scores(attribution_score)

			GNNgraph.label = original_label

			output[label] = attribution_score
		output_for_metrics_calculation.append(output)

	execution_time = sum(tmp_timing_list)/(len(tmp_timing_list))

	# Obtain attribution score for use in generating saliency map for comparison with zero tensors
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

	elif interpretability_config["number_of_samples"] > 0:
		# Randomly sample from existing list:
		graph_idxes = list(range(len(output_for_metrics_calculation)))
		random.shuffle(graph_idxes)
		output_for_generating_saliency_map.update({"layergradcam_%s_%s" % (str(assign_type), str(label)): []
												   for _, label in dataset_features["label_dict"].items()})

		# Begin appending found samples
		for index in graph_idxes:
			tmp_label = output_for_metrics_calculation[index]['graph'].label
			element_name = "layergradcam_%s_%s" % (str(assign_type), str(tmp_label))
			if len(output_for_generating_saliency_map[element_name]) < interpretability_config["number_of_samples"]:
				output_for_generating_saliency_map[element_name].append(
					(output_for_metrics_calculation[index]['graph'], output_for_metrics_calculation[index][tmp_label]))

	return output_for_metrics_calculation, output_for_generating_saliency_map, execution_time
