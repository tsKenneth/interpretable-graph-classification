# LEGACY CODE: This was made in response to writing custom backward hooks for SortPooling layer.
# However, it was found that backward for topk is already handled in the native autograd

from captum.attr import DeepLift
import torch

# Check if module backward hook can safely be used for the module that produced
# this inputs / outputs mapping
def _check_valid_module(inputs, outputs):
	curr_fn = outputs.grad_fn
	first_next = curr_fn.next_functions[0]
	try:
		return first_next[0] == inputs[first_next[1]].grad_fn
	except IndexError:
		return False

class DeepLiftGraph(DeepLift):
	def __init__(self, model):
		super().__init__(model)

	def _forward_pooling_hook(self, module, inputs, outputs):
		k_attr_name = "k"
		num_nodes_attr_name = "num_nodes"

		setattr(module, k_attr_name, outputs[1])
		setattr(module, num_nodes_attr_name, inputs[0].size()[0])

		module.unsorted_grad = None

		def tensor_backward_hook(grad):
			return module.unsorted_grad

		inputs[0].register_hook(tensor_backward_hook)

	def _backward_pooling_hook(self, module, grad_input, grad_output, eps=1e-10):
		grad_input_extracted = grad_input[1]
		grad_input_unsorted = torch.zeros(
			[module.num_nodes, grad_input_extracted.size()[1]],
			dtype=torch.float32)

		print(module.k)

		# Un-sort gradient input with k_indices obtained
		i = 0
		for index in module.k:
			grad_input_unsorted[index] = grad_input_extracted[i]
			i+=1

		# To be overriden in tensor hook
		module.unsorted_grad = grad_input_unsorted

		print("Backward Hook End")
		return grad_input

	def _register_pooling_hooks(self, module, input_type="non_ref"):
		if input_type != "ref":
			forward_handle = module.register_forward_hook(self._forward_pooling_hook)
			backward_handle = module.register_backward_hook(self._backward_pooling_hook)
			self.forward_handles.append(forward_handle)
			self.backward_handles.append(backward_handle)

	def _register_hooks(self, module, input_type="non_ref"):
		module_fullname = str(type(module))
		has_already_hooks = len(module._backward_hooks) > 0

		if "pooling.SortPooling" in module_fullname:
			self._register_pooling_hooks(module, input_type)
			return
		elif (
				"nn.modules.container" in module_fullname
				or has_already_hooks
				or not self._is_non_linear(module)
		):
			return

		# adds forward hook to leaf nodes that are non-linear
		if input_type != "ref":
			forward_handle = module.register_forward_hook(self._forward_hook)
			backward_handle = module.register_backward_hook(self._backward_hook)
			self.forward_handles.append(forward_handle)
			self.backward_handles.append(backward_handle)
		else:
			handle = module.register_forward_hook(self._forward_hook_ref)
			ref_handle = "ref_handle"
			setattr(module, ref_handle, handle)
