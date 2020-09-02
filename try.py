import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn as nn


# # x = np.arange(0, 1, 0.01)
# # x = torch.tensor(x)
# x = torch.tensor([[0, 0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8, 0.9]])
# labels = torch.tensor([[0, 0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 1, 1]])
# # y = 2 ** torch.round(torch.log2(x + 0.00001))
# # plt.plot(x, y)
# # plt.show()
# print(x.size())
#
# predictions = (x >= 0.5).float()
# is_equal = (predictions == labels).float()
# accuracy = torch.sum(is_equal.reshape(-1)) / (is_equal.reshape(-1).size(0))
# print('accuracy:', accuracy)  # 0.9907 ?

# class Model(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(Model, self).__init__()
#
#         self.encoder = nn.sequential(
#             nn.Con
#
#         )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
#
#
# x = torch.randn(1, 3, 480, 480)
# model = Model()
