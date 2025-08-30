"""Modified from https://github.com/alexmonti19/dagnet"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Intent-aware GAT layer with separate intent attention head.
    Based on https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha, concat=True, intent_embed_dim=32):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.intent_embed_dim = intent_embed_dim

        # Original spatial attention parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Intent attention head: learns attention from intent embeddings
        self.intent_attention = nn.Linear(2 * intent_embed_dim, 1)
        nn.init.xavier_uniform_(self.intent_attention.weight, gain=1.414)
        
        # Learnable mixing parameter for spatial vs intent attention
        self.intent_beta = nn.Parameter(torch.tensor(0.1))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, intent_embeds=None):
#         print(input.shape,self.W.shape)
        h = torch.mm(input, self.W)  # matrix multiplication of the matrices
        N = h.size()[0]
        
        # Compute spatial attention (original GAT mechanism)
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        spatial_e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Compute intent-based attention using separate attention head
        intent_e = torch.zeros_like(spatial_e)
        if intent_embeds is not None:
            # Create pairwise intent embedding concatenations for all agent pairs
            for i in range(N):
                for j in range(N):
                    intent_pair = torch.cat([intent_embeds[i], intent_embeds[j]], dim=0)  # [2 * intent_embed_dim]
                    intent_e[i, j] = self.intent_attention(intent_pair).squeeze()
        
        # Combine spatial and intent attention
        total_e = spatial_e + self.intent_beta * intent_e
        
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