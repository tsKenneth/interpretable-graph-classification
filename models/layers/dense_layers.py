import torch
import torch.nn as nn
from models.layers.lib.layer_util import gnn_spmm

class DenseLayers(nn.Module):
	def __init__(self,
				 input_dim,
				 output_dim,
				 latent_dim=[50, 50, 50]):
		super(DenseLayers, self).__init__()

		if len(latent_dim) == 0:
			self.prediction_model = nn.Linear(input_dim, output_dim)
		else:
			dense_layers = []
			for hidden_dim in latent_dim:
				dense_layers.append(nn.Linear(input_dim, hidden_dim))
				dense_layers.append(nn.ReLU())
				input_dim = hidden_dim
			dense_layers.append(nn.Linear(hidden_dim, output_dim))
			self.prediction_model = nn.Sequential(*dense_layers)

	def forward(self, input_tensor):
		return self.prediction_model(input_tensor)
