# import torch
# import torchvision

# -*- coding: utf-8 -*-
"""utils.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GHuj_7cof-C3BMqNOx7tQ5nmIPD3Rebz
"""

'''
  returns a list of string, it contains the name of parent folder of the given file
'''
def get_parent(o):
  return [o.parent.name]

'''
  returns a number signifying the accuracy of the inputs based of the multiple targets
'''
def accuracy_multi_class(inp, targ, thresh=0.5, sigmoid=True):
  if sigmoid: inp.sigmoid()
  return ((inp >= thresh) == targ.bool()).float().mean()

'''
  returns the loss based on the inputs and target
'''
def binary_cross_entropy_loss(inputs, targets):
  inputs.sigmoid()
  return -torch.where(targets==1, inputs, 1-inputs).log().mean()