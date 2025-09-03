#!/usr/bin/env python3
import os

import torch
import torch.nn as nn
import torch.utils.cpp_extension as cpp_extension

import _depthwise_conv2d_implicit_gemm_C as _extension
from depthwise_conv2d_implicit_gemm import _DepthWiseConv2dImplicitGEMMFP32, _DepthWiseConv2dImplicitGEMMFP16


__all__ = ["DepthWiseConv2dImplicitGEMM"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# base 口\\\口
class DepthWiseSharing_v3(nn.Module):
    def __init__(self, in_channels, kernel_size, stride = 1, padding = 1, groups = 1, bias=True):
        super(DepthWiseSharing_v3, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        if bias:
            self.bias =  nn.Parameter(torch.randn(in_channels).to(device))
        else:
            self.bias = None

        self.left_unitary_matrix = nn.Parameter(torch.randn(1, 1, kernel_size, 5).to(device))
        self.right_unitary_matrix = nn.Parameter(torch.randn(1, 1, 5, kernel_size).to(device))
        self.left_scalar = nn.Parameter(torch.randn(in_channels, 1, 1, 5).to(device))
        self.right_scalar = nn.Parameter(torch.randn(in_channels, 1, 5, 1).to(device))
        self.lambda_scalar_matrix = nn.Parameter(torch.randn(in_channels, 1, 1, 5).to(device))


    def forward(self, input):
        self.left_unitary_matrix.repeat(1, self.in_channels, 1, 1)
        self.right_unitary_matrix.repeat(1, self.in_channels, 1, 1)

        weight = (self.left_unitary_matrix * self.left_scalar) * self.lambda_scalar_matrix @ (self.right_scalar * self.right_unitary_matrix)
        if input.dtype == torch.float32:
            input = _DepthWiseConv2dImplicitGEMMFP32.apply(input, weight)
        elif input.dtype == torch.float16:
            input = _DepthWiseConv2dImplicitGEMMFP16.apply(input, weight)
        else:
            raise TypeError("Only support fp32 and fp16, get {}".format(input.dtype))
        if self.bias is not None:
            input = input + self.bias.to(input).view(1, -1, 1, 1)
        return input



# large \口\\\口\
class DepthWiseSharing_v4(nn.Module):
    def __init__(self, in_channels, kernel_size, stride = 1, padding = 1, groups = 1, bias=True):
        super(DepthWiseSharing_v4, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        if bias:
            self.bias =  nn.Parameter(torch.randn(in_channels).to(device))
        else:
            self.bias = None

        self.left_unitary_matrix = nn.Parameter(torch.randn(1, 1, kernel_size, 5).to(device))
        self.right_unitary_matrix = nn.Parameter(torch.randn(1, 1, 5, kernel_size).to(device))
        self.left_scalar = nn.Parameter(torch.randn(in_channels, 1, 1, 5).to(device))
        self.right_scalar = nn.Parameter(torch.randn(in_channels, 1, 5, 1).to(device))
        self.left_left_scalar = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1).to(device))
        self.right_right_scalar = nn.Parameter(torch.randn(in_channels, 1, 1, kernel_size).to(device))
        self.lambda_scalar_matrix = nn.Parameter(torch.randn(in_channels, 1, 1, 5).to(device))


    def forward(self, input):
        self.left_unitary_matrix.repeat(1, self.in_channels, 1, 1)
        self.right_unitary_matrix.repeat(1, self.in_channels, 1, 1)

        weight = (self.left_left_scalar * self.left_unitary_matrix * self.left_scalar) * self.lambda_scalar_matrix @ (self.right_scalar * self.right_unitary_matrix * self.right_right_scalar)
        if input.dtype == torch.float32:
            input = _DepthWiseConv2dImplicitGEMMFP32.apply(input, weight)
        elif input.dtype == torch.float16:
            input = _DepthWiseConv2dImplicitGEMMFP16.apply(input, weight)
        else:
            raise TypeError("Only support fp32 and fp16, get {}".format(input.dtype))
        if self.bias is not None:
            input = input + self.bias.to(input).view(1, -1, 1, 1)
        return input

    
# Small 口\口
class DepthWiseSharing_v5(nn.Module):
    def __init__(self, in_channels, kernel_size, stride = 1, padding = 1, groups = 1, bias=True):
        super(DepthWiseSharing_v5, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        if bias:
            self.bias =  nn.Parameter(torch.randn(in_channels).to(device))
        else:
            self.bias = None

        self.left_unitary_matrix = nn.Parameter(torch.randn(1, 1, kernel_size, 5).to(device))
        self.right_unitary_matrix = nn.Parameter(torch.randn(1, 1, 5, kernel_size).to(device))
        self.lambda_scalar_matrix = nn.Parameter(torch.randn(in_channels, 1, 1, 5).to(device))


    def forward(self, input):
        self.left_unitary_matrix.repeat(1, self.in_channels, 1, 1)
        self.right_unitary_matrix.repeat(1, self.in_channels, 1, 1)

        weight = self.left_unitary_matrix * self.lambda_scalar_matrix @ self.right_unitary_matrix
        if input.dtype == torch.float32:
            input = _DepthWiseConv2dImplicitGEMMFP32.apply(input, weight)
        elif input.dtype == torch.float16:
            input = _DepthWiseConv2dImplicitGEMMFP16.apply(input, weight)
        else:
            raise TypeError("Only support fp32 and fp16, get {}".format(input.dtype))
        if self.bias is not None:
            input = input + self.bias.to(input).view(1, -1, 1, 1)
        return input


 

#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------


# base
class PointWiseSharing_v5_1(nn.Module):
    def __init__(self, in_channels, dw_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True, R=200):
        super().__init__()
        self.in_channels = in_channels
        self.dw_channels = dw_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.R = R

        if bias:
            self.bias = nn.Parameter(torch.randn(dw_channels).to(device))
        else:
            self.bias = None

        self.left = nn.Parameter(torch.randn(in_channels, R))
        self.right = nn.Parameter(torch.randn(R, in_channels))

        self.lambda_scalar_list_left = []
        self.lambda_scalar_list_right = []
        self.lambda_scalar_list = []
        for _ in range(dw_channels//in_channels):
            self.lambda_scalar_list_left.append(nn.Parameter(torch.randn(1, R)))
            self.lambda_scalar_list_right.append(nn.Parameter(torch.randn(R, 1)))
            self.lambda_scalar_list.append(nn.Parameter(torch.randn(1, R)))
        self.lambda_scalar_list_left = nn.ParameterList(self.lambda_scalar_list_left)
        self.lambda_scalar_list_right = nn.ParameterList(self.lambda_scalar_list_right)
        self.lambda_scalar_list = nn.ParameterList(self.lambda_scalar_list)

    def forward(self, x):
        # 将ParameterList转换为张量，增加批量维度
        # 形状: (num_groups, 1, R)
        lambda_left = torch.stack(list(self.lambda_scalar_list_left), dim=0)
        # 形状: (num_groups, R, 1)
        lambda_right = torch.stack(list(self.lambda_scalar_list_right), dim=0)
        # 形状: (num_groups, 1, R)
        lambda_scalar = torch.stack(list(self.lambda_scalar_list), dim=0)
        
        # 向量化计算所有分组的矩阵
        # 形状: (num_groups, in_channels, R)
        left_part = self.left * lambda_left
        # 形状: (num_groups, in_channels, R)
        left_scaled = left_part * lambda_scalar
        # 形状: (num_groups, R, in_channels)
        right_part = lambda_right * self.right
        
        # 批量矩阵乘法: (num_groups, in_channels, R) @ (num_groups, R, in_channels)
        # 结果形状: (num_groups, in_channels, in_channels)
        pointwise_matrices = torch.bmm(left_scaled, right_part)
        
        # 转换为卷积权重格式
        weight = pointwise_matrices.view(-1, self.in_channels).unsqueeze(2).unsqueeze(3)

        return nn.functional.conv2d(
            x, 
            weight, 
            padding=self.padding, 
            stride=self.stride, 
            bias=self.bias, 
            groups=self.groups
        )

class PointWiseSharing_v5_2(nn.Module):
    def __init__(self, dw_channels, in_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True, R=200):
        super().__init__()
        self.dw_channels = dw_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups 
        self.padding = padding
        self.R = R

        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels).to(device))
            
        else:
            self.bias = None
        
        self.left = nn.Parameter(torch.randn(in_channels, R))
        self.right = nn.Parameter(torch.randn(R, in_channels))

        self.lambda_scalar_list_left = []
        self.lambda_scalar_list_right = []
        self.lambda_scalar_list = []
        for _ in range(dw_channels//in_channels):
            self.lambda_scalar_list_left.append(nn.Parameter(torch.randn(1, R)))
            self.lambda_scalar_list_right.append(nn.Parameter(torch.randn(R, 1)))
            self.lambda_scalar_list.append(nn.Parameter(torch.randn(1, R)))
        self.lambda_scalar_list_left = nn.ParameterList(self.lambda_scalar_list_left)
        self.lambda_scalar_list_right = nn.ParameterList(self.lambda_scalar_list_right)
        self.lambda_scalar_list = nn.ParameterList(self.lambda_scalar_list)

    def forward(self, x):
        # 将ParameterList转换为张量，增加批量维度
        # 形状: (num_groups, 1, R)
        lambda_left = torch.stack(list(self.lambda_scalar_list_left), dim=0)
        # 形状: (num_groups, R, 1)
        lambda_right = torch.stack(list(self.lambda_scalar_list_right), dim=0)
        # 形状: (num_groups, 1, R)
        lambda_scalar = torch.stack(list(self.lambda_scalar_list), dim=0)
        
        # 向量化计算所有分组的矩阵
        # 形状: (num_groups, in_channels, R)
        left_part = self.left * lambda_left
        # 形状: (num_groups, in_channels, R)
        left_scaled = left_part * lambda_scalar
        # 形状: (num_groups, R, in_channels)
        right_part = lambda_right * self.right
        
        # 批量矩阵乘法: (num_groups, in_channels, R) @ (num_groups, R, in_channels)
        # 结果形状: (num_groups, in_channels, in_channels)
        pointwise_matrices = torch.bmm(left_scaled, right_part)
        
        # 按dim=1拼接（与原版本一致）并转换为卷积权重格式
        # 先调整维度顺序，再合并，最后增加卷积核维度
        weight = pointwise_matrices.transpose(0, 1).contiguous().view(self.out_channels, -1)\
                                  .unsqueeze(2).unsqueeze(3)

        return nn.functional.conv2d(
            x, 
            weight, 
            padding=self.padding, 
            stride=self.stride, 
            bias=self.bias, 
            groups=self.groups
        )
    

# large 
class PointWiseSharing_v6_1(nn.Module):
    def __init__(self, in_channels, dw_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True, R=200):
        super().__init__()
        self.in_channels = in_channels
        self.dw_channels = dw_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.R = R

        if bias:
            self.bias = nn.Parameter(torch.randn(dw_channels).to(device))
        else:
            self.bias = None

        self.left = nn.Parameter(torch.randn(in_channels, R))
        self.right = nn.Parameter(torch.randn(R, in_channels))

        self.lambda_scalar_list_left = []
        self.lambda_scalar_list_right = []
        self.lambda_scalar_list_left_left = []
        self.lambda_scalar_list_right_right = []
        self.lambda_scalar_list = []
        for _ in range(dw_channels//in_channels):
            self.lambda_scalar_list_left.append(nn.Parameter(torch.randn(1, R)))
            self.lambda_scalar_list_right.append(nn.Parameter(torch.randn(R, 1)))
            self.lambda_scalar_list_left_left.append(nn.Parameter(torch.randn(in_channels, 1)))
            self.lambda_scalar_list_right_right.append(nn.Parameter(torch.randn(1, in_channels)))
            self.lambda_scalar_list.append(nn.Parameter(torch.randn(1, R)))
        self.lambda_scalar_list_left = nn.ParameterList(self.lambda_scalar_list_left)
        self.lambda_scalar_list_right = nn.ParameterList(self.lambda_scalar_list_right)
        self.lambda_scalar_list_left_left = nn.ParameterList(self.lambda_scalar_list_left_left)
        self.lambda_scalar_list_right_right = nn.ParameterList(self.lambda_scalar_list_right_right)
        self.lambda_scalar_list = nn.ParameterList(self.lambda_scalar_list)

    def forward(self, x):

        kernel_list = []

        for (lambda_scalar_left_left, lambda_scalar_right_right, lambda_scalar_left, lambda_scalar_right, lambda_scalar) in zip(
            self.lambda_scalar_list_left_left, self.lambda_scalar_list_right_right, self.lambda_scalar_list_left, self.lambda_scalar_list_right, self.lambda_scalar_list):
            pointwise_matrix = (lambda_scalar_left_left * self.left * lambda_scalar_left) * lambda_scalar @ (lambda_scalar_right * self.right * lambda_scalar_right_right)
            kernel_list.append(pointwise_matrix)

        weight = torch.cat(kernel_list, dim=0).unsqueeze(2).unsqueeze(3)

        return nn.functional.conv2d(x, weight, padding = self.padding, stride = self.stride, 
                                    bias = self.bias, groups=self.groups)

class PointWiseSharing_v6_2(nn.Module):
    def __init__(self, dw_channels, in_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True, R=200):
        super().__init__()
        self.dw_channels = dw_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups 
        self.padding = padding
        self.R = R

        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels).to(device))
        else:
            self.bias = None
        
        self.left = nn.Parameter(torch.randn(in_channels, R))
        self.right = nn.Parameter(torch.randn(R, in_channels))

        self.lambda_scalar_list_left = []
        self.lambda_scalar_list_right = []
        self.lambda_scalar_list_left_left = []
        self.lambda_scalar_list_right_right = []
        self.lambda_scalar_list = []
        for _ in range(dw_channels//in_channels):
            self.lambda_scalar_list_left.append(nn.Parameter(torch.randn(1, R)))
            self.lambda_scalar_list_right.append(nn.Parameter(torch.randn(R, 1)))
            self.lambda_scalar_list_left_left.append(nn.Parameter(torch.randn(in_channels, 1)))
            self.lambda_scalar_list_right_right.append(nn.Parameter(torch.randn(1, in_channels)))
            self.lambda_scalar_list.append(nn.Parameter(torch.randn(1, R)))
        self.lambda_scalar_list_left = nn.ParameterList(self.lambda_scalar_list_left)
        self.lambda_scalar_list_right = nn.ParameterList(self.lambda_scalar_list_right)
        self.lambda_scalar_list_left_left = nn.ParameterList(self.lambda_scalar_list_left_left)
        self.lambda_scalar_list_right_right = nn.ParameterList(self.lambda_scalar_list_right_right)
        self.lambda_scalar_list = nn.ParameterList(self.lambda_scalar_list)

    def forward(self, x):

        kernel_list = []

        for (lambda_scalar_left_left, lambda_scalar_right_right, lambda_scalar_left, lambda_scalar_right, lambda_scalar) in zip(
            self.lambda_scalar_list_left_left, self.lambda_scalar_list_right_right, self.lambda_scalar_list_left, self.lambda_scalar_list_right, self.lambda_scalar_list):
            pointwise_matrix = (lambda_scalar_left_left * self.left * lambda_scalar_left) * lambda_scalar @ (lambda_scalar_right * self.right * lambda_scalar_right_right)
            kernel_list.append(pointwise_matrix)

        weight = torch.cat(kernel_list, dim=1).unsqueeze(2).unsqueeze(3)

        return nn.functional.conv2d(x, weight, padding = self.padding, stride = self.stride, 
                                    bias = self.bias, groups=self.groups)



# small
class PointWiseSharing_v7_1(nn.Module):
    def __init__(self, in_channels, dw_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True, R=200):
        super().__init__()
        self.in_channels = in_channels
        self.dw_channels = dw_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.R = R

        if bias:
            self.bias = nn.Parameter(torch.randn(dw_channels).to(device))
        else:
            self.bias = None

        self.left = nn.Parameter(torch.randn(in_channels, R))
        self.right = nn.Parameter(torch.randn(R, in_channels))

        self.lambda_scalar_list = []
        for _ in range(dw_channels//in_channels):
            self.lambda_scalar_list.append(nn.Parameter(torch.randn(1, R)))
        self.lambda_scalar_list = nn.ParameterList(self.lambda_scalar_list)

    def forward(self, x):

        kernel_list = []

        for lambda_scalar in self.lambda_scalar_list:
            pointwise_matrix = self.left * lambda_scalar @ self.right
            kernel_list.append(pointwise_matrix)

        weight = torch.cat(kernel_list, dim=0).unsqueeze(2).unsqueeze(3)

        return nn.functional.conv2d(x, weight, padding = self.padding, stride = self.stride, 
                                    bias = self.bias, groups=self.groups)

class PointWiseSharing_v7_2(nn.Module):
    def __init__(self, dw_channels, in_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True, R=200):
        super().__init__()
        self.dw_channels = dw_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups 
        self.padding = padding
        self.R = R

        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels).to(device))
        else:
            self.bias = None
        
        self.left = nn.Parameter(torch.randn(in_channels, R))
        self.right = nn.Parameter(torch.randn(R, in_channels))

        self.lambda_scalar_list = []
        for _ in range(dw_channels//in_channels):
            self.lambda_scalar_list.append(nn.Parameter(torch.randn(1, R)))
        self.lambda_scalar_list = nn.ParameterList(self.lambda_scalar_list)

    def forward(self, x):

        kernel_list = []

        for lambda_scalar in self.lambda_scalar_list:
            pointwise_matrix = self.left * lambda_scalar @ self.right
            kernel_list.append(pointwise_matrix)

        weight = torch.cat(kernel_list, dim=1).unsqueeze(2).unsqueeze(3)

        return nn.functional.conv2d(x, weight, padding = self.padding, stride = self.stride, 
                                    bias = self.bias, groups=self.groups)