"""Modified from https://github.com/alexmonti19/dagnet"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Multi-head GAT layer with specialized spatial and intent attention heads.
    Based on https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha, concat=True, intent_embed_dim=32, head_type='spatial'):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.intent_embed_dim = intent_embed_dim
        self.head_type = head_type  # 'spatial' or 'intent'

        # Common transformation matrix
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        if head_type == 'spatial':
            # Spatial-only attention (original GAT)
            self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
            nn.init.xavier_uniform_(self.a.data, gain=1.414)
        elif head_type == 'intent':
            # Intent-specialized attention: processes spatial + intent features
            self.a = nn.Parameter(torch.zeros(size=(2 * out_features + 2 * intent_embed_dim, 1)))
            nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, intent_embeds=None):
#         print(input.shape,self.W.shape)
        h = torch.mm(input, self.W)  # matrix multiplication of the matrices
        N = h.size()[0]
        
        if self.head_type == 'spatial':
            # Spatial-only attention (original GAT behavior)
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
            
        elif self.head_type == 'intent':
            # Intent-specialized attention: combine spatial features with intent embeddings
            if intent_embeds is not None:
                # Create spatial + intent feature pairs for all agent combinations
                spatial_pairs = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
                
                # Create intent embedding pairs
                intent_pairs = torch.zeros(N, N, 2 * self.intent_embed_dim, device=h.device)
                for i in range(N):
                    for j in range(N):
                        intent_pairs[i, j] = torch.cat([intent_embeds[i], intent_embeds[j]], dim=0)
                
                # Concatenate spatial and intent features
                a_input = torch.cat([spatial_pairs, intent_pairs], dim=2)
            else:
                # Fall back to spatial-only if no intent embeddings provided
                a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
                # Pad with zeros for intent part
                zero_intent = torch.zeros(N, N, 2 * self.intent_embed_dim, device=h.device)
                a_input = torch.cat([a_input, zero_intent], dim=2)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = e  # torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=0)
        h_prime = torch.matmul(attention, h)
        # print("attn_",attention)
        # print("h",h)
        # print("h_",h_prime)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime