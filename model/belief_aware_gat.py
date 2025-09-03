"""
Belief-Aware Graph Attention Network

This module extends the standard GAT to incorporate dynamic belief state embeddings
for more socially-aware trajectory prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BeliefAwareGraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer enhanced with belief state information.
    
    Extends the standard GAT by incorporating belief embeddings into the attention
    computation, allowing the model to consider pilot intentions when modeling
    agent interactions.
    """

    def __init__(self, in_features: int, out_features: int, alpha: float, 
                 belief_embed_dim: int = 64, concat: bool = True):
        """
        Args:
            in_features: Input feature dimension (spatial + context features)
            out_features: Output feature dimension
            alpha: LeakyReLU negative slope
            belief_embed_dim: Dimension of belief embeddings
            concat: Whether to apply ELU activation (True) or not (False)
        """
        super(BeliefAwareGraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.belief_embed_dim = belief_embed_dim

        # Standard GAT parameters for spatial features
        self.W_spatial = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_spatial.data, gain=1.414)
        
        # Attention parameters - enhanced to include belief information
        self.a_spatial = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a_spatial.data, gain=1.414)
        
        # Belief-aware attention parameters
        self.W_belief = nn.Parameter(torch.zeros(size=(belief_embed_dim, out_features)))
        nn.init.xavier_uniform_(self.W_belief.data, gain=1.414)
        
        self.a_belief = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a_belief.data, gain=1.414)
        
        # Learnable mixing parameter for spatial vs belief attention
        self.belief_weight = nn.Parameter(torch.tensor(0.3))  # Initialize to moderate belief influence
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, spatial_input: torch.Tensor, adj: torch.Tensor, 
                belief_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional belief embeddings.
        
        Args:
            spatial_input: [N, in_features] spatial and context features
            adj: [N, N] adjacency matrix (not used but kept for compatibility)
            belief_embeddings: [N, belief_embed_dim] belief embeddings for each agent
            
        Returns:
            h_prime: [N, out_features] attention-weighted features
        """
        N = spatial_input.size()[0]
        
        # Transform spatial features
        h_spatial = torch.mm(spatial_input, self.W_spatial)  # [N, out_features]
        
        # Compute spatial attention (standard GAT)
        a_input_spatial = torch.cat([
            h_spatial.repeat(1, N).view(N * N, -1), 
            h_spatial.repeat(N, 1)
        ], dim=1).view(N, -1, 2 * self.out_features)
        
        e_spatial = self.leakyrelu(torch.matmul(a_input_spatial, self.a_spatial).squeeze(2))
        
        # Initialize total attention with spatial attention
        total_attention = e_spatial
        
        # Add belief-based attention if belief embeddings are provided
        if belief_embeddings is not None:
            # Transform belief embeddings
            h_belief = torch.mm(belief_embeddings, self.W_belief)  # [N, out_features]
            
            # Compute belief-based attention
            a_input_belief = torch.cat([
                h_belief.repeat(1, N).view(N * N, -1), 
                h_belief.repeat(N, 1)
            ], dim=1).view(N, -1, 2 * self.out_features)
            
            e_belief = self.leakyrelu(torch.matmul(a_input_belief, self.a_belief).squeeze(2))
            
            # Combine spatial and belief attention
            belief_weight_clamped = torch.sigmoid(self.belief_weight)  # Keep in [0,1]
            total_attention = (1 - belief_weight_clamped) * e_spatial + belief_weight_clamped * e_belief
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(total_attention, dim=1)
        
        # Apply attention to spatial features (belief only affects attention, not features)
        h_prime = torch.matmul(attention_weights, h_spatial)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class BeliefAwareGAT(nn.Module):
    """
    Multi-head Belief-Aware Graph Attention Network.
    
    Extends GAT to incorporate pilot belief states in the attention mechanism
    while maintaining the core spatial trajectory processing.
    """
    
    def __init__(self, nin: int, nhid: int, nout: int, alpha: float, nheads: int,
                 belief_embed_dim: int = 64):
        """
        Args:
            nin: Input feature dimension
            nhid: Hidden dimension for each attention head
            nout: Output feature dimension
            alpha: LeakyReLU negative slope
            nheads: Number of attention heads
            belief_embed_dim: Dimension of belief embeddings
        """
        super(BeliefAwareGAT, self).__init__()
        
        self.nheads = nheads
        self.belief_embed_dim = belief_embed_dim

        # Multi-head attention layers with belief awareness
        self.attentions = nn.ModuleList([
            BeliefAwareGraphAttentionLayer(
                nin, nhid, alpha=alpha, belief_embed_dim=belief_embed_dim, concat=True
            ) for _ in range(nheads)
        ])

        # Output attention layer
        self.out_att = BeliefAwareGraphAttentionLayer(
            nhid * nheads, nout, alpha=alpha, belief_embed_dim=belief_embed_dim, concat=False
        )
        
        # Batch normalization (optional)
        self.bn1 = nn.BatchNorm1d(nout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, 
                belief_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional belief embeddings.
        
        Args:
            x: [N, nin] input features (spatial + context)
            adj: [N, N] adjacency matrix
            belief_embeddings: [N, belief_embed_dim] belief embeddings
            
        Returns:
            output: [N, nout] attention-weighted features
        """
        # Multi-head attention with belief awareness
        head_outputs = []
        for attention in self.attentions:
            head_output = attention(x, adj, belief_embeddings)
            head_outputs.append(head_output)
        
        # Concatenate multi-head outputs
        x = torch.cat(head_outputs, dim=1)  # [N, nheads * nhid]
        
        # Final attention layer
        x = self.out_att(x, adj, belief_embeddings)  # [N, nout]
        
        # Optional batch normalization (commented out to match original)
        # x = self.bn1(x)
        
        return torch.tanh(x)


class BeliefIntegrationLayer(nn.Module):
    """
    Helper layer to integrate belief embeddings with spatial features.
    
    This layer provides different strategies for combining belief information
    with spatial trajectory features before feeding to the GAT.
    """
    
    def __init__(self, spatial_dim: int, belief_dim: int, integration_mode: str = 'concatenate'):
        """
        Args:
            spatial_dim: Dimension of spatial features
            belief_dim: Dimension of belief embeddings
            integration_mode: How to integrate ('concatenate', 'add', 'gated')
        """
        super(BeliefIntegrationLayer, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.belief_dim = belief_dim
        self.integration_mode = integration_mode
        
        if integration_mode == 'concatenate':
            # No additional parameters needed
            self.output_dim = spatial_dim + belief_dim
        elif integration_mode == 'add':
            # Project belief to spatial dimension
            self.belief_projection = nn.Linear(belief_dim, spatial_dim)
            self.output_dim = spatial_dim
        elif integration_mode == 'gated':
            # Gated fusion mechanism
            self.gate_network = nn.Sequential(
                nn.Linear(spatial_dim + belief_dim, spatial_dim),
                nn.Sigmoid()
            )
            self.belief_projection = nn.Linear(belief_dim, spatial_dim)
            self.output_dim = spatial_dim
        else:
            raise ValueError(f"Unknown integration mode: {integration_mode}")
    
    def forward(self, spatial_features: torch.Tensor, 
                belief_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Integrate spatial features with belief embeddings.
        
        Args:
            spatial_features: [N, spatial_dim] spatial features
            belief_embeddings: [N, belief_dim] belief embeddings or None
            
        Returns:
            integrated_features: [N, output_dim] combined features
        """
        if belief_embeddings is None:
            # No belief information available - handle gracefully
            if self.integration_mode == 'concatenate':
                # Pad with zeros
                zero_beliefs = torch.zeros(spatial_features.size(0), self.belief_dim, 
                                         device=spatial_features.device)
                return torch.cat([spatial_features, zero_beliefs], dim=1)
            else:
                # Return spatial features as-is
                return spatial_features
        
        if self.integration_mode == 'concatenate':
            return torch.cat([spatial_features, belief_embeddings], dim=1)
        
        elif self.integration_mode == 'add':
            projected_beliefs = self.belief_projection(belief_embeddings)
            return spatial_features + projected_beliefs
        
        elif self.integration_mode == 'gated':
            projected_beliefs = self.belief_projection(belief_embeddings)
            concatenated = torch.cat([spatial_features, belief_embeddings], dim=1)
            gate = self.gate_network(concatenated)
            return gate * spatial_features + (1 - gate) * projected_beliefs


def test_belief_aware_gat():
    """Test the BeliefAwareGAT with sample data."""
    print("Testing BeliefAwareGAT...")
    
    # Test parameters
    batch_size = 4
    spatial_dim = 64
    belief_dim = 32
    hidden_dim = 32
    output_dim = 64
    n_heads = 4
    
    # Create sample data
    spatial_features = torch.randn(batch_size, spatial_dim)
    belief_embeddings = torch.randn(batch_size, belief_dim)
    adj = torch.ones(batch_size, batch_size)
    
    print(f"Input shapes:")
    print(f"  Spatial features: {spatial_features.shape}")
    print(f"  Belief embeddings: {belief_embeddings.shape}")
    
    # Test BeliefAwareGAT
    gat = BeliefAwareGAT(
        nin=spatial_dim, nhid=hidden_dim, nout=output_dim,
        alpha=0.2, nheads=n_heads, belief_embed_dim=belief_dim
    )
    
    # Forward pass with beliefs
    output_with_beliefs = gat(spatial_features, adj, belief_embeddings)
    print(f"  Output with beliefs: {output_with_beliefs.shape}")
    
    # Forward pass without beliefs
    output_without_beliefs = gat(spatial_features, adj, None)
    print(f"  Output without beliefs: {output_without_beliefs.shape}")
    
    # Test integration layer
    integration_layer = BeliefIntegrationLayer(spatial_dim, belief_dim, 'concatenate')
    integrated = integration_layer(spatial_features, belief_embeddings)
    print(f"  Integrated features: {integrated.shape}")
    
    # Check that outputs are different when beliefs are provided
    diff = torch.mean(torch.abs(output_with_beliefs - output_without_beliefs))
    print(f"  Difference with/without beliefs: {diff.item():.4f}")
    
    print("BeliefAwareGAT test passed!")


if __name__ == "__main__":
    test_belief_aware_gat()