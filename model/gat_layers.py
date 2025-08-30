"""Modified from https://github.com/alexmonti19/dagnet"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Intent-aware GAT layer with learnable intent interaction matrix.
    Based on https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha, concat=True, num_intent_classes=16):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_intent_classes = num_intent_classes

        # Original spatial attention parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Intent interaction matrix: learned 16x16 matrix for intent pair interactions
        self.intent_interaction = nn.Parameter(torch.randn(num_intent_classes, num_intent_classes) * 0.1)
        # Intent attention mixing parameter
        self.intent_alpha = nn.Parameter(torch.tensor(0.1))  # Learnable mixing weight

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, intent_labels=None):
#         print(input.shape,self.W.shape)
        h = torch.mm(input, self.W)  # matrix multiplication of the matrices
        N = h.size()[0]
        
        # Compute spatial attention (original GAT mechanism)
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        spatial_e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Compute intent-based attention bias if intent labels provided
        intent_bias = torch.zeros_like(spatial_e)
        if intent_labels is not None:
            # Convert intent labels to indices (assuming labels are 1-16, convert to 0-15)
            intent_indices = (intent_labels - 1).clamp(0, self.num_intent_classes - 1)
            
            # Create intent interaction matrix for all agent pairs
            for i in range(N):
                for j in range(N):
                    intent_i = intent_indices[i]
                    intent_j = intent_indices[j]
                    intent_bias[i, j] = self.intent_interaction[intent_i, intent_j]
        
        # Combine spatial and intent attention
        total_e = spatial_e + self.intent_alpha * intent_bias
        
        # Apply softmax and compute output
        zero_vec = -9e15 * torch.ones_like(total_e)
        attention = total_e  # torch.where(adj > 0, total_e, zero_vec)
        attention = F.softmax(attention, dim=0)
        h_prime = torch.matmul(attention, h)
        # print("attn_",attention)
        # print("h",h)
        # print("h_",h_prime)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime