import torch
import copy 

input_relu = torch.rand(5) 
sorted, _ = input_relu.sort()
input_relu = input_relu - sorted[1]
input_relu.requires_grad = True
input_softmax = copy.deepcopy(input_relu)
input_softmax_jac = copy.deepcopy(input_relu)

relu = torch.nn.functional.relu
softmax = torch.nn.functional.softmax

relu_out = relu(input_relu)
grad_relu = torch.autograd.grad(relu_out, input_relu, grad_outputs=torch.ones_like(relu_out), create_graph=True)[0]

softmax_out = softmax(input_softmax)
grad_softmax = torch.autograd.grad(softmax_out, input_softmax, grad_outputs=torch.ones_like(softmax_out), create_graph=True)[0]
jac_softmax = torch.autograd.functional.jacobian(softmax, input_softmax_jac, create_graph=True)

print(input_relu)
print(input_softmax)
print(relu_out)
print(grad_relu)
print(softmax_out)
print(grad_softmax)
print(jac_softmax)