from __future__ import print_function

import os
import sys
import numpy as np
import torch
import torch.nn as nn

class SortPooling(nn.Module):
	def __init__(self, sort_pooling_k, total_latent_dim):
		super(SortPooling, self).__init__()
		self.sort_pooling_k, = sort_pooling_k,
		self.total_latent_dim = total_latent_dim

	def forward(self, concat_pool, subg, graph_sizes):
		sort_channel = concat_pool[:, -1]
		batch_sortpooling_graphs = torch.zeros(len(graph_sizes),
											   self.sort_pooling_k, self.total_latent_dim)

		# Check if we can use CUDA
		if torch.cuda.is_available() and isinstance(sort_channel, torch.cuda.FloatTensor):
			cuda_flag = True
		else:
			cuda_flag = False

		if cuda_flag:
			batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()

		# Sortpooling
		accum_count = 0
		for i in range(subg):
			to_sort = sort_channel[accum_count: accum_count + graph_sizes[i]]
			k = self.sort_pooling_k if self.sort_pooling_k <= graph_sizes[i] else graph_sizes[i]
			_, topk_indices = to_sort.topk(k)
			topk_indices += accum_count
			sortpooling_graph = concat_pool.index_select(0, topk_indices)

			if k < self.sort_pooling_k:
				to_pad = torch.zeros(self.sort_pooling_k - k, self.total_latent_dim)
				if cuda_flag:
					to_pad = to_pad.cuda()

				sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
			batch_sortpooling_graphs[i] = sortpooling_graph

			accum_count += graph_sizes[i]

		return batch_sortpooling_graphs, topk_indices