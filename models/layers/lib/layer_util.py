import torch
import torch.nn as nn
from torch.autograd import Variable

class MySpMM(torch.autograd.Function):
	@staticmethod
	def forward(ctx, sp_mat, dense_mat):
		ctx.save_for_backward(sp_mat, dense_mat)
		return torch.mm(sp_mat, dense_mat)

	@staticmethod
	def backward(ctx, grad_output):
		sp_mat, dense_mat = ctx.saved_variables
		grad_matrix1 = grad_matrix2 = None

		assert not ctx.needs_input_grad[0]
		if ctx.needs_input_grad[1]:
			grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))

		return grad_matrix1, grad_matrix2


def gnn_spmm(sp_mat, dense_mat):
	return MySpMM.apply(sp_mat, dense_mat)
