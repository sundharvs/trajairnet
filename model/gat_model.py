"""Modified from https://github.com/alexmonti19/dagnet"""

import torch
import torch.nn as nn
from model.gat_layers import GraphAttentionLayer


class GAT(nn.Module):
    """Intent-aware GAT with learnable intent interaction matrix."""
    def __init__(self, nin, nhid, nout, alpha, nheads, num_intent_classes=16):
        super(GAT, self).__init__()

        self.attentions = [GraphAttentionLayer(nin, nhid, alpha=alpha, concat=True, num_intent_classes=num_intent_classes) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nout, alpha=alpha, concat=False, num_intent_classes=num_intent_classes)
        self.bn1 = nn.BatchNorm1d(nout)

    def forward(self, x, adj, intent_labels=None):
        # Multi-head attention with intent awareness
        x = torch.cat([att(x, adj, intent_labels) for att in self.attentions], dim=1)
        # Output attention with intent awareness
        x = self.out_att(x, adj, intent_labels)
        return torch.tanh(x)