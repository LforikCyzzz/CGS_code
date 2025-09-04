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
class DepthWiseSharing_Base(nn.Module):
    def __init__(self, in_channels, kernel_size, stride = 1, padding = 1, groups = 1, bias=True):
        super(DepthWiseSharing_Base, self).__init__()
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
class DepthWiseSharing_Large(nn.Module):
    def __init__(self, in_channels, kernel_size, stride = 1, padding = 1, groups = 1, bias=True):
        super(DepthWiseSharing_Large, self).__init__()
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
class DepthWiseSharing_Small(nn.Module):
    def __init__(self, in_channels, kernel_size, stride = 1, padding = 1, groups = 1, bias=True):
        super(DepthWiseSharing_Small, self).__init__()
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

# base 口\\\口
class PointWiseSharing_Base_1(nn.Module):
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
        lambda_left = torch.stack(list(self.lambda_scalar_list_left), dim=0)
        lambda_right = torch.stack(list(self.lambda_scalar_list_right), dim=0)
        lambda_scalar = torch.stack(list(self.lambda_scalar_list), dim=0)
        
        left_part = self.left * lambda_left
        left_scaled = left_part * lambda_scalar
        right_part = lambda_right * self.right

        pointwise_matrices = torch.bmm(left_scaled, right_part)
        
        weight = pointwise_matrices.view(-1, self.in_channels).unsqueeze(2).unsqueeze(3)

        return nn.functional.conv2d(
            x, 
            weight, 
            padding=self.padding, 
            stride=self.stride, 
            bias=self.bias, 
            groups=self.groups
        )

class PointWiseSharing_Base_2(nn.Module):
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
        lambda_left = torch.stack(list(self.lambda_scalar_list_left), dim=0)
        lambda_right = torch.stack(list(self.lambda_scalar_list_right), dim=0)
        lambda_scalar = torch.stack(list(self.lambda_scalar_list), dim=0)

        left_part = self.left * lambda_left
        left_scaled = left_part * lambda_scalar
        right_part = lambda_right * self.right
        
        pointwise_matrices = torch.bmm(left_scaled, right_part)
        
        
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
    

# large \口\\\口\
class PointWiseSharing_Large_1(nn.Module):
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
        lambda_left_left = torch.stack(list(self.lambda_scalar_list_left_left), dim=0)
        lambda_right_right = torch.stack(list(self.lambda_scalar_list_right_right), dim=0)
        lambda_left = torch.stack(list(self.lambda_scalar_list_left), dim=0)
        lambda_right = torch.stack(list(self.lambda_scalar_list_right), dim=0)
        lambda_scalar = torch.stack(list(self.lambda_scalar_list), dim=0)
        
        left_part = lambda_left_left * self.left * lambda_left
        left_scaled = left_part * lambda_scalar
        right_part = lambda_right * self.right * lambda_right_right

        pointwise_matrices = torch.bmm(left_scaled, right_part)
        
        weight = pointwise_matrices.view(-1, self.in_channels).unsqueeze(2).unsqueeze(3)

        return nn.functional.conv2d(
            x, 
            weight, 
            padding=self.padding, 
            stride=self.stride, 
            bias=self.bias, 
            groups=self.groups
        )

class PointWiseSharing_Large_2(nn.Module):
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
        lambda_left_left = torch.stack(list(self.lambda_scalar_list_left_left), dim=0)
        lambda_right_right = torch.stack(list(self.lambda_scalar_list_right_right), dim=0)
        lambda_left = torch.stack(list(self.lambda_scalar_list_left), dim=0)
        lambda_right = torch.stack(list(self.lambda_scalar_list_right), dim=0)
        lambda_scalar = torch.stack(list(self.lambda_scalar_list), dim=0)

        left_part = lambda_left_left * self.left * lambda_left
        left_scaled = left_part * lambda_scalar
        right_part = lambda_right * self.right * lambda_right_right
        
        pointwise_matrices = torch.bmm(left_scaled, right_part)
        
        
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


# Small 口\口
class PointWiseSharing_Small_1(nn.Module):
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
        lambda_scalar = torch.stack(list(self.lambda_scalar_list), dim=0)
        D = lambda_scalar.shape[0]
        
        left_scaled = self.left * lambda_scalar
        right_part = self.right.unsqueeze(0).expand(D, -1, -1)
        
        pointwise_matrices = torch.bmm(left_scaled, right_part)
        weight = pointwise_matrices.view(-1, self.in_channels).unsqueeze(2).unsqueeze(3)
        
        return nn.functional.conv2d(
            x, 
            weight, 
            padding=self.padding, 
            stride=self.stride, 
            bias=self.bias, 
            groups=self.groups
        )

class PointWiseSharing_Small_2(nn.Module):
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
        lambda_scalar = torch.stack(list(self.lambda_scalar_list), dim=0)
        D = lambda_scalar.shape[0]
        
        left_scaled = self.left * lambda_scalar
        right_part = self.right.unsqueeze(0).expand(D, -1, -1)
        
        pointwise_matrices = torch.bmm(left_scaled, right_part)
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