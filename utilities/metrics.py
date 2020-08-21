import numpy
import torch

import torch.nn.functional as F
from copy import deepcopy
from sklearn import metrics

from utilities.util import graph_to_tensor, hamming

def auc_scores(all_targets, all_scores):
	all_scores = torch.cat(all_scores).cpu().numpy()
	number_of_classes = int(all_scores.shape[1])

	# For binary classification:
	roc_auc = 0.0
	prc_auc = 0.0
	if number_of_classes == 2:
		# Take only second column (i.e. scores for positive label)
		all_scores = all_scores[:, 1]
		roc_auc = metrics.roc_auc_score(
			all_targets, all_scores, average='macro')
		prc_auc = metrics.average_precision_score(
			all_targets, all_scores, average='macro', pos_label=1)
	# For multi-class classification:
	if number_of_classes > 2:
		# Hand & Till (2001) implementation (ovo)
		roc_auc = metrics.roc_auc_score(
			all_targets, all_scores, multi_class='ovo', average='macro')

		# TODO: build PRC-AUC calculations for multi-class datasets

	return roc_auc, prc_auc

# Fidelity ====================================================================
def get_accuracy(trained_classifier_model, GNNgraph_list, dataset_features, cuda):
	trained_classifier_model.eval()
	true_equal_pred_pairs = []

	# Instead of sending the whole list as batch,
	# do it one by one in case classifier do not support batch-processing
	# TODO: Enable batch processing support
	for GNNgraph in GNNgraph_list:
		node_feat, n2n, subg = graph_to_tensor(
            [GNNgraph], dataset_features["feat_dim"],
            dataset_features["edge_feat_dim"], cuda)

		output = trained_classifier_model(node_feat, n2n, subg, [GNNgraph])
		logits = F.log_softmax(output, dim=1)
		pred = logits.data.max(1, keepdim=True)[1]

		if GNNgraph.label == int(pred[0]):
			true_equal_pred_pairs.append(1)
		else:
			true_equal_pred_pairs.append(0)

	return sum(true_equal_pred_pairs)/len(true_equal_pred_pairs)

def get_roc_auc(trained_classifier_model, GNNgraph_list, dataset_features, cuda):
	trained_classifier_model.eval()
	score_list = []
	target_list = []

	if dataset_features["num_class"] > 2:
		print("Unable to calculate fidelity for multiclass datset")
		return 0

	# Instead of sending the whole list as batch,
	# do it one by one in case classifier do not support batch-processing
	# TODO: Enable batch processing support
	for GNNgraph in GNNgraph_list:
		node_feat, n2n, subg = graph_to_tensor(
            [GNNgraph], dataset_features["feat_dim"],
            dataset_features["edge_feat_dim"], cuda)

		output = trained_classifier_model(node_feat, n2n, subg, [GNNgraph])
		logits = F.log_softmax(output, dim=1)
		prob = F.softmax(logits, dim=1)

		score_list.append(prob.cpu().detach())
		target_list.append(GNNgraph.label)

	score_list = torch.cat(score_list).cpu().numpy()
	score_list = score_list[:, 1]

	roc_auc = metrics.roc_auc_score(
		target_list, score_list, average='macro')

	return roc_auc

def is_salient(score, importance_range):
	start, end = importance_range
	if start <= score <= end:
		return True
	else:
		return False

def occlude_graphs(metric_attribution_scores, dataset_features, importance_range):
	# Transform the graphs, occlude nodes with significant attribution scores
	occluded_GNNgraph_list = []
	for group in metric_attribution_scores:
		GNNgraph = deepcopy(group['graph'])
		attribution_score = group[GNNgraph.label]

		# Go through every node in graph to check if node is salient
		for i in range(len(attribution_score)):
			# Only occlude nodes that provide significant positive contribution
			if is_salient((attribution_score[i]), importance_range):
				# Occlude node by assigning it an "UNKNOWN" label
				if dataset_features['have_node_labels'] is True:
					GNNgraph.node_labels[i] = None
				if dataset_features['have_node_attributions'] is True:
					GNNgraph.node_features[i].fill(0)

		occluded_GNNgraph_list.append(GNNgraph)
	return occluded_GNNgraph_list


def get_fidelity(trained_classifier_model, metric_attribution_scores, dataset_features, config, cuda):
	importance_range = config["metrics"]["fidelity"]["importance_range"].split(",")
	importance_range = [float(bound) for bound in importance_range]

	GNNgraph_list = [group["graph"] for group in metric_attribution_scores]

	roc_auc_prior_occlusion = get_roc_auc(trained_classifier_model, GNNgraph_list, dataset_features, cuda)
	occluded_GNNgraph_list = occlude_graphs(metric_attribution_scores, dataset_features, importance_range)
	roc_auc_after_occlusion = get_roc_auc(trained_classifier_model, occluded_GNNgraph_list, dataset_features, cuda)

	fidelity_score = roc_auc_prior_occlusion - roc_auc_after_occlusion
	return fidelity_score

# Contrastivity ====================================================================
def binarize_score_list(attribution_scores_list, importance_range):
	binarized_scores_list = []
	for scores in attribution_scores_list:
		binary_score = ''
		for score in scores:
			if is_salient(abs(float(score)), importance_range):
				binary_score += '1'
			else:
				binary_score += '0'
		binarized_scores_list.append(binary_score)
	return binarized_scores_list

def get_contrastivity(metric_attribution_scores, dataset_features, config):
	importance_range = config["metrics"]["fidelity"]["importance_range"].split(",")
	importance_range = [float(bound) for bound in importance_range]

	# Binarize score list according to their saliency
	class_0_binarized_scores_list = binarize_score_list(
		[group[0] for group in metric_attribution_scores], importance_range)
	class_1_binarized_scores_list = binarize_score_list(
		[group[1] for group in metric_attribution_scores], importance_range)

	result_list = []
	# Calculate hamming distance
	for class_0, class_1 in zip(class_0_binarized_scores_list, class_1_binarized_scores_list):
		assert len(class_0) == len(class_1)
		d = hamming(class_0, class_1)
		result_list.append(d / len(class_0))

	return sum(result_list) / len(result_list)

# Sparsity ====================================================================
def count_salient_nodes(attribution_scores_list, important_range):
	salient_node_count_list = []
	for scores in attribution_scores_list:
		count = 0
		for score in scores:
			if is_salient(abs(float(score)), important_range):
				count += 1
		salient_node_count_list.append(count)
	return salient_node_count_list

def get_sparsity(metric_attribution_scores, config):
	importance_range = config["metrics"]["fidelity"]["importance_range"].split(",")
	importance_range = [float(bound) for bound in importance_range]

	class_0_significant_nodes_count = count_salient_nodes(
		[group[0] for group in metric_attribution_scores], importance_range)
	class_1_significant_nodes_count = count_salient_nodes(
		[group[1] for group in metric_attribution_scores], importance_range)
	graphs_number_of_nodes = [group['graph'].number_of_nodes for group in metric_attribution_scores]

	# measure the average sparsity score across all samples
	result_list = []
	for i in range(len(graphs_number_of_nodes)):
		d = class_0_significant_nodes_count[i] + \
			class_1_significant_nodes_count[i]
		d /= (graphs_number_of_nodes[i] * 2)
		result_list.append(1 - d)
	return sum(result_list) / len(result_list)

def compute_metric(trained_classifier_model, metric_attribution_scores, dataset_features, config, cuda):
	if config["metrics"]["fidelity"]["enabled"] is True:
		fidelity_metric = get_fidelity(trained_classifier_model, metric_attribution_scores, dataset_features,
						config, cuda)
	else:
		fidelity_metric = 0

	if config["metrics"]["contrastivity"]["enabled"] is True:
		contrastivity_metric = get_contrastivity(metric_attribution_scores, dataset_features, config)
	else:
		contrastivity_metric = 0

	if config["metrics"]["sparsity"]["enabled"] is True:
		sparsity_metric = get_sparsity(metric_attribution_scores, config)
	else:
		sparsity_metric = 0

	return fidelity_metric, contrastivity_metric, sparsity_metric
