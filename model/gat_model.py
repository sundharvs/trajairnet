"""Modified from https://github.com/alexmonti19/dagnet"""

import torch
import torch.nn as nn
from model.gat_layers import GraphAttentionLayer


class GAT(nn.Module):
    """Multi-head GAT with specialized spatial and intent attention heads."""
    def __init__(self, nin, nhid, nout, alpha, nheads, intent_embed_dim=32, intent_head_ratio=0.25):
        super(GAT, self).__init__()
        
        # Split heads between spatial and intent processing
        self.num_intent_heads = max(1, int(nheads * intent_head_ratio))
        self.num_spatial_heads = nheads - self.num_intent_heads
        
        # Create spatial-specialized heads
        self.spatial_attentions = [GraphAttentionLayer(nin, nhid, alpha=alpha, concat=True, 
                                                     intent_embed_dim=intent_embed_dim, head_type='spatial') 
                                 for _ in range(self.num_spatial_heads)]
        for i, attention in enumerate(self.spatial_attentions):
            self.add_module('spatial_attention_{}'.format(i), attention)
        
        # Create intent-specialized heads
        self.intent_attentions = [GraphAttentionLayer(nin, nhid, alpha=alpha, concat=True, 
                                                    intent_embed_dim=intent_embed_dim, head_type='intent') 
                                for _ in range(self.num_intent_heads)]
        for i, attention in enumerate(self.intent_attentions):
            self.add_module('intent_attention_{}'.format(i), attention)

        # Output attention layer (spatial-only for consistency)
        self.out_att = GraphAttentionLayer(nhid * nheads, nout, alpha=alpha, concat=False, 
                                         intent_embed_dim=intent_embed_dim, head_type='spatial')
        self.bn1 = nn.BatchNorm1d(nout)

    def forward(self, x, adj, intent_embeds=None):
        # Process spatial-specialized heads
        spatial_outputs = [att(x, adj) for att in self.spatial_attentions]
        
        # Process intent-specialized heads
        intent_outputs = [att(x, adj, intent_embeds) for att in self.intent_attentions]
        
        # Concatenate all head outputs
        x = torch.cat(spatial_outputs + intent_outputs, dim=1)
        
        # Final output attention
        x = self.out_att(x, adj)
        return torch.tanh(x)