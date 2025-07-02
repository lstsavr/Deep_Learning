import torch
import torch.nn as nn
import torch.nn.functional as F

Class MyDenseLayer(nn.module):
  def __init__(self, input_dim, output_dim):
    super().__init__():

    self.W = nn.Parameter(torch.randn(input_dim, output_dim, requires_grad=True)
    self.b = nn.Parameter(torch.randn(1, output_dim, requires_grad=True)
  
  def forward(self,input)
    z = torch.matmul(inputs, self.W) + self.b
    output = torch.sigmoid(z)
    return output
