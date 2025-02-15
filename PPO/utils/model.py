import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


# Creating a masking matrix
def masking_matrix(m, n, p):
    """
    Creates an m x n binary matrix with exactly p fraction of zeros in each row.

    Args:
        m (int): Number of rows.
        n (int): Number of columns.
        p (float): Fraction of ones in each row (e.g., 0.3 means 30% zeros).

    Returns:
        torch.Tensor: An m x n matrix of 0s and 1s.
    """

    # Compute the number of zeros in each row.
    # We use round() to get as close as possible to the desired fraction.
    num_zeros = int(round((1-p) * m))
    # Make sure we do not have more zeros than the row length.
    num_zeros = min(num_zeros, m)

    # Create an m x n tensor initialized with zeros.
    matrix = torch.zeros(n, m, dtype=torch.int)
    for i in range(n):
        # Get a random permutation of column indices
        perm = torch.randperm(m)
        # Select indices that will become ones (the remaining ones after the zeros)
        one_indices = perm[num_zeros:]
        # Set those indices in the i-th row to 1.
        matrix[i, one_indices] = 1
    return matrix


# Feature Scaler
class FeatureScaler(nn.Module):
    def __init__(self, hidden_size=0, adjacency_matrix=0):
        super().__init__()

        # Masking matrix
        #F = self.masking_matrix(1024, 8, p=0.5)

        # GNN matrix
        #F = torch.FloatTensor(self.gnn_normalize(adjacency_matrix))

        # Scaler vector
        scale = torch.FloatTensor([np.sqrt(2/np.pi)])
        F = torch.randn(hidden_size)/scale
        b = torch.randn(hidden_size) / scale

        # Register constant vector or matrix into the buffer
        self.register_buffer("F", F)
        self.register_buffer("b", b)

    def gnn_normalize(self, adjacency_matrix):
        # Add self-loops (optional, common in GNNs)
        adjacency_matrix = adjacency_matrix + np.eye(adjacency_matrix.shape[0])
        # Compute the degree matrix
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        # Compute D^(-1/2)
        degree_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
        # Compute the normalized adjacency matrix
        normalized_adj = degree_inv_sqrt @ adjacency_matrix @ degree_inv_sqrt
        return normalized_adj

    def forward(self, x):
        return x * self.F + self.b


# Customized linear weight matrix to mask
class CustomLinear(nn.Module):
    """
    A linear layer that applies a mask to its weight matrix before
    performing the linear transformation.
    """

    def __init__(self, in_features, out_features, p, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        mask = masking_matrix(in_features, out_features, p)
        # Register the mask as a buffer so it is moved to GPU if model.cuda() is called,
        # but does not count as a trainable parameter.
        self.register_buffer('mask', mask)

        # Create the usual weight and (optionally) bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # You can use any initialization you like here.
        # For example, kaiming_uniform_ is a common choice:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if self.bias is not None:
            # PyTorch's recommended uniform initialization for bias
            fan_in = self.in_features
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # element-wise multiply the weight by the mask
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)


# Customized convolution weight matrix to mask
class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.use_bias = True
        else:
            self.use_bias = False

        # Initialize filters (weights) and biases
        self.register_buffer(
            "scale_factors",
            torch.randn(out_channels, 1, 1, 1) * torch.tensor([np.sqrt(2 / np.pi)], dtype=torch.float32)
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # Apply padding
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        # Unfold the input to extract patches
        unfolded = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)

        # Reshape filters to match unfolded input
        weight_matrix = self.weights * self.scale_factors
        weight_matrix = weight_matrix.view(self.out_channels, -1)

        # Perform matrix multiplication
        conv_out = weight_matrix @ unfolded  # (out_channels, num_patches * batch_size)
        conv_out = conv_out.view(batch_size, self.out_channels, -1)

        # Reshape back to image shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        conv_out = conv_out.view(batch_size, self.out_channels, out_height, out_width)

        # Add bias
        if self.use_bias:
            conv_out += self.bias.view(1, self.out_channels, 1, 1)

        return conv_out
