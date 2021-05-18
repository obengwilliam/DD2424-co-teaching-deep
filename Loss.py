import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import bottleneck as bn


# Loss functions
# Input:
#       y_1    Prediction from Net1
#       y_2    Prediction from Net2
#       t      Target TrainData
#       R_rate Remember Rate
#       ind    index
# Output:
#       loss_1  exchanged loss from N2 to N1
#       loss_2  exchanged loss from N1 to N2
def loss_coteaching(y_1, y_2, t, R_rate):
    # Compute the number of data remembered
    num_rem = int(len(t) * R_rate)

    # Calculate the loss of Network 1 and pick up the R_rate% smallest ones' index
    loss_1 = F.cross_entropy(y_1, t, reduce=False) # Ignore the mean loss and return the loss of each element
    loss1_sorted_ind = bn.argpartition(loss_1.data, kth=num_rem)[:num_rem]  # .cuda()

    # Calculate the loss of Network 2 and rank
    loss_2 = F.cross_entropy(y_2, t, reduce=False)
    loss2_sorted_ind = bn.argpartition(loss_2.data, kth=num_rem)[:num_rem]  # .cuda()

    # Exchange the index and use the other network's sample to calculate the loss
    loss_1_exchanged = F.cross_entropy(y_1[loss2_sorted_ind], t[loss2_sorted_ind])
    loss_2_exchanged = F.cross_entropy(y_2[loss1_sorted_ind, t[loss1_sorted_ind]])

    # Return the exchanged loss for back propagation
    return torch.sum(loss_1_exchanged)/num_rem, torch.sum(loss_2_exchanged)/num_rem
