import torch
import numpy as np
import torch.nn as nn


class RGBgradients(nn.Module):
    def __init__(self, weight):
        super().__init__()
        k_height, k_width = weight.shape[1:]

        padding_x = int((k_height-1)/2)
        padding_y = int((k_width-1)/2)

        self.conv = nn.Conv2d(3, 6, (k_height, k_width), bias=False,
                              padding=(padding_x, padding_y))

        weight1x = np.array([weight[0],
                             np.zeros((k_height, k_width)),
                             np.zeros((k_height, k_width))])

        weight1y = np.array([weight[1],
                             np.zeros((k_height, k_width)),
                             np.zeros((k_height, k_width))])

        weight2x = np.array([np.zeros((k_height, k_width)),
                             weight[0],
                             np.zeros((k_height, k_width))])

        weight2y = np.array([np.zeros((k_height, k_width)),
                             weight[1],
                             np.zeros((k_height, k_width))])

        weight3x = np.array([np.zeros((k_height, k_width)),
                             np.zeros((k_height, k_width)),
                             weight[0]])
        weight3y = np.array([np.zeros((k_height, k_width)),
                             np.zeros((k_height, k_width)),
                             weight[1]])

        weight_final = torch.from_numpy(np.array([weight1x, weight1y,
                                                  weight2x, weight2y,
                                                  weight3x, weight3y])).type(torch.FloatTensor)

        if self.conv.weight.shape == weight_final.shape:
            self.conv.weight = nn.Parameter(weight_final)
            self.conv.weight.requires_grad_(False)
        else:
            print('Error: The shape of the given weights is not correct')

    def forward(self, x):
        return self.conv(x)
