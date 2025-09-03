"""
Belief-Aware TrajAirNet

This module extends TrajAirNet to incorporate dynamic belief states from radio
communications for enhanced socially-aware trajectory prediction.
"""

import torch
from torch import nn
from typing import Optional, List

from model.tcn_model import TemporalConvNet
from model.belief_aware_gat import BeliefAwareGAT, BeliefIntegrationLayer
from model.cvae_base import CVAE
from model.utils import acc_to_abs
from model.belief_states import BeliefEncoder


class BeliefAwareTrajAirNet(nn.Module):
    """
    Enhanced TrajAirNet with dynamic belief state integration.
    
    Architecture:
    1. TCN encoders for trajectory features
    2. Context CNN for environmental features  
    3. BeliefEncoder for pilot intent sequences
    4. BeliefAwareGAT for socially-aware attention
    5. CVAE for probabilistic trajectory prediction
    """
    
    def __init__(self, args):
        super(BeliefAwareTrajAirNet, self).__init__()

        # Model dimensions
        input_size = args.input_channels
        n_classes = int(args.preds / args.preds_step)
        num_channels = [args.tcn_channel_size] * args.tcn_layers
        num_channels.append(n_classes)
        tcn_kernel_size = args.tcn_kernels
        dropout = args.dropout
        
        # Belief embedding configuration
        belief_embed_dim = getattr(args, 'belief_embed_dim', 64)
        belief_vocab_size = getattr(args, 'belief_vocab_size', 35)  # From belief_states.py
        belief_integration_mode = getattr(args, 'belief_integration_mode', 'concatenate')
        
        # GAT dimensions - updated to include belief embeddings
        graph_hidden = args.graph_hidden
        context_dim = args.num_context_output_c
        
        # Calculate GAT input dimension based on integration mode
        spatial_dim = n_classes * args.obs + context_dim
        
        if belief_integration_mode == 'concatenate':
            gat_in = spatial_dim + belief_embed_dim
        else:
            gat_in = spatial_dim  # Other modes keep same dimension
        
        gat_out = gat_in  # Keep same output dimension
        n_heads = args.gat_heads
        alpha = args.alpha
        
        # CVAE dimensions
        cvae_encoder = [n_classes * n_classes]
        for layer in range(args.cvae_layers):
            cvae_encoder.append(args.cvae_channel_size)
        cvae_decoder = [args.cvae_channel_size] * args.cvae_layers
        cvae_decoder.append(input_size * args.mlp_layer)

        # Initialize components
        
        # Trajectory encoders (same as original)
        self.tcn_encoder_x = TemporalConvNet(
            input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout
        )
        self.tcn_encoder_y = TemporalConvNet(
            input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout
        )
        
        # Context processing (same as original)
        self.context_conv = nn.Conv1d(
            in_channels=args.num_context_input_c, out_channels=1, 
            kernel_size=args.cnn_kernels
        )
        self.context_linear = nn.Linear(args.obs - 1, context_dim)
        
        # Belief state processing (NEW)
        self.belief_encoder = BeliefEncoder(
            vocab_size=belief_vocab_size,
            embed_dim=belief_embed_dim,
            hidden_dim=belief_embed_dim,
            num_layers=2
        )
        
        # Feature integration (NEW)
        self.belief_integration = BeliefIntegrationLayer(
            spatial_dim=spatial_dim,
            belief_dim=belief_embed_dim,
            integration_mode=belief_integration_mode
        )
        
        # Belief-aware GAT (MODIFIED)
        self.gat = BeliefAwareGAT(
            nin=gat_in, nhid=graph_hidden, nout=gat_out, 
            alpha=alpha, nheads=n_heads, belief_embed_dim=belief_embed_dim
        )
        
        # CVAE and decoder (same as original)
        self.cvae = CVAE(
            encoder_layer_sizes=cvae_encoder,
            latent_size=args.cvae_hidden,
            decoder_layer_sizes=cvae_decoder,
            conditional=True,
            num_labels=gat_out + gat_in
        )
        self.linear_decoder = nn.Linear(args.mlp_layer, n_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        self.linear_decoder.weight.data.normal_(0, 0.05)
        self.context_linear.weight.data.normal_(0, 0.05)
        self.context_conv.weight.data.normal_(0, 0.1)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, adj: torch.Tensor, 
                context: torch.Tensor, belief_sequences: Optional[List[List[int]]] = None,
                belief_lengths: Optional[torch.Tensor] = None, sort: bool = False):
        """
        Forward pass with optional belief sequences.
        
        Args:
            x: [seq_len, batch, num_agents] observed trajectories
            y: [seq_len, batch, num_agents] future trajectories
            adj: [num_agents, num_agents] adjacency matrix
            context: [seq_len, batch, num_agents] context features
            belief_sequences: List of belief sequences for each agent
            belief_lengths: [num_agents] lengths of belief sequences
            sort: Whether to sort (unused, kept for compatibility)
            
        Returns:
            recon_y: List of reconstructed trajectories for each agent
            m: List of means for each agent
            var: List of log variances for each agent
        """
        num_agents = x.shape[2]
        
        # Encode trajectories and context (same as original)
        encoded_trajectories_x = []
        encoded_appended_trajectories_x = []
        encoded_trajectories_y = []
        
        for agent in range(num_agents):
            # Encode observed trajectory
            x1 = torch.transpose(x[:, :, agent][None, :, :], 1, 2)
            encoded_x = self.tcn_encoder_x(x1)
            encoded_x = torch.flatten(encoded_x)[None, None, :]
            encoded_trajectories_x.append(encoded_x)
            
            # Encode context
            c1 = torch.transpose(context[:, :, agent][None, :, :], 1, 2)
            encoded_context = self.context_conv(c1)
            encoded_context = self.relu(self.context_linear(encoded_context))
            
            # Combine trajectory and context features
            appended_x = torch.cat((encoded_x, encoded_context), dim=2)
            encoded_appended_trajectories_x.append(appended_x)
            
            # Encode future trajectory
            y1 = torch.transpose(y[:, :, agent][None, :, :], 1, 2)
            encoded_y = self.tcn_encoder_y(y1)
            encoded_y = torch.flatten(encoded_y)[None, None, :]
            encoded_trajectories_y.append(encoded_y)
        
        # Prepare spatial features for GAT
        spatial_features = torch.squeeze(torch.stack(encoded_appended_trajectories_x, dim=2))
        if len(spatial_features.shape) == 1:
            spatial_features = torch.unsqueeze(spatial_features, dim=0)
        
        # Process belief sequences (NEW)
        belief_embeddings = None
        if belief_sequences is not None and belief_lengths is not None:
            # Convert belief sequences to tensors and encode
            from model.belief_states import pad_belief_sequences
            
            padded_beliefs, lengths = pad_belief_sequences(belief_sequences)
            belief_embeddings = self.belief_encoder(padded_beliefs, lengths)
        
        # Integrate belief embeddings with spatial features
        if self.belief_integration.integration_mode == 'concatenate':
            # Concatenation mode: integrate before GAT
            integrated_features = self.belief_integration(spatial_features, belief_embeddings)
            gat_output = self.gat(integrated_features, adj, belief_embeddings)
        else:
            # Other modes: use belief embeddings directly in GAT attention
            gat_output = self.gat(spatial_features, adj, belief_embeddings)
        
        # Decode trajectories (same as original)
        recon_y = []
        m = []
        var = []
        
        for agent in range(num_agents):
            # Prepare features for CVAE
            H_x = gat_output[agent].unsqueeze(0).unsqueeze(0)
            H_xx = encoded_appended_trajectories_x[agent]
            H_x = torch.cat((H_xx, H_x), dim=2)
            
            H_y = encoded_trajectories_y[agent]
            H_yy, means, log_var, z = self.cvae(H_y, H_x)
            
            # Decode trajectory
            H_yy = torch.reshape(H_yy, (3, -1))
            recon_y_x = self.linear_decoder(H_yy)
            recon_y_x = torch.unsqueeze(recon_y_x, dim=0)
            recon_y_x = acc_to_abs(recon_y_x, x[:, :, agent][:, :, None])
            
            recon_y.append(recon_y_x)
            m.append(means)
            var.append(log_var)
            
        return recon_y, m, var
    
    def inference(self, x: torch.Tensor, z: torch.Tensor, adj: torch.Tensor, 
                  context: torch.Tensor, belief_sequences: Optional[List[List[int]]] = None,
                  belief_lengths: Optional[torch.Tensor] = None):
        """
        Inference with optional belief sequences.
        
        Args:
            x: [seq_len, batch, num_agents] observed trajectories
            z: [1, 1, latent_size] latent variable
            adj: [num_agents, num_agents] adjacency matrix
            context: [seq_len, batch, num_agents] context features
            belief_sequences: List of belief sequences for each agent
            belief_lengths: [num_agents] lengths of belief sequences
            
        Returns:
            recon_y: List of predicted trajectories for each agent
        """
        num_agents = x.shape[2]
        
        # Encode trajectories and context
        encoded_trajectories_x = []
        encoded_appended_trajectories_x = []
        
        for agent in range(num_agents):
            # Encode trajectory and context
            x1 = torch.transpose(x[:, :, agent][None, :, :], 1, 2)
            c1 = torch.transpose(context[:, :, agent][None, :, :], 1, 2)
            encoded_context = self.context_conv(c1)
            encoded_context = self.relu(self.context_linear(encoded_context))
            
            encoded_x = self.tcn_encoder_x(x1)
            encoded_x = torch.flatten(encoded_x)[None, None, :]
            encoded_trajectories_x.append(encoded_x)
            
            appended_x = torch.cat((encoded_x, encoded_context), dim=2)
            encoded_appended_trajectories_x.append(appended_x)
        
        # Prepare features and process beliefs
        spatial_features = torch.squeeze(torch.stack(encoded_appended_trajectories_x, dim=2))
        if len(spatial_features.shape) == 1:
            spatial_features = torch.unsqueeze(spatial_features, dim=0)
        
        # Process beliefs
        belief_embeddings = None
        if belief_sequences is not None and belief_lengths is not None:
            from model.belief_states import pad_belief_sequences
            padded_beliefs, lengths = pad_belief_sequences(belief_sequences)
            belief_embeddings = self.belief_encoder(padded_beliefs, lengths)
        
        # GAT with beliefs
        if self.belief_integration.integration_mode == 'concatenate':
            integrated_features = self.belief_integration(spatial_features, belief_embeddings)
            gat_output = self.gat(integrated_features, adj, belief_embeddings)
        else:
            gat_output = self.gat(spatial_features, adj, belief_embeddings)
        
        # Generate predictions
        recon_y = []
        
        for agent in range(num_agents):
            H_x = (gat_output[agent].unsqueeze(0)).unsqueeze(0)
            H_xx = encoded_appended_trajectories_x[agent]
            H_x = torch.cat((H_xx, H_x), dim=2)
            H_yy = self.cvae.inference(z, H_x)
            H_yy = torch.reshape(H_yy, (3, -1))
            
            recon_y_x = self.linear_decoder(H_yy)
            recon_y_x = torch.unsqueeze(recon_y_x, dim=0)
            recon_y_x = acc_to_abs(recon_y_x, x[:, :, agent][:, :, None])
            
            recon_y.append(recon_y_x.squeeze().detach())
     
        return recon_y


def test_belief_trajairnet():
    """Test BeliefAwareTrajAirNet with sample data."""
    print("Testing BeliefAwareTrajAirNet...")
    
    # Mock args object
    class Args:
        input_channels = 3
        preds = 120
        preds_step = 10
        tcn_channel_size = 64
        tcn_layers = 2
        tcn_kernels = 4
        dropout = 0.05
        num_context_output_c = 7
        obs = 11
        graph_hidden = 64
        gat_heads = 4
        alpha = 0.2
        cvae_layers = 2
        cvae_channel_size = 32
        cvae_hidden = 32
        mlp_layer = 16
        num_context_input_c = 2
        cnn_kernels = 2
        belief_embed_dim = 32
        belief_vocab_size = 35
        belief_integration_mode = 'concatenate'
    
    args = Args()
    
    # Create model
    model = BeliefAwareTrajAirNet(args)
    print(f"Model created successfully")
    
    # Create sample data
    batch_size = 1
    num_agents = 3
    obs_len = args.obs
    pred_len = int(args.preds / args.preds_step)
    
    x = torch.randn(obs_len, batch_size, num_agents)
    y = torch.randn(pred_len, batch_size, num_agents)
    context = torch.randn(obs_len, batch_size, num_agents)
    adj = torch.ones(num_agents, num_agents)
    
    # Sample belief sequences
    belief_sequences = [
        [4, 6, 8, 12],    # downwind_8, base_8, final_8, land_8
        [5, 7, 9, 13],    # downwind_26, base_26, final_26, land_26
        [29]              # unknown
    ]
    belief_lengths = torch.tensor([4, 4, 1])
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  y: {y.shape}")
    print(f"  context: {context.shape}")
    print(f"  belief_sequences: {[len(seq) for seq in belief_sequences]}")
    
    # Test forward pass with beliefs
    recon_y, m, var = model(x, y, adj, context, belief_sequences, belief_lengths)
    print(f"Forward pass with beliefs successful")
    print(f"  Number of reconstructed trajectories: {len(recon_y)}")
    print(f"  Reconstruction shape: {recon_y[0].shape}")
    
    # Test forward pass without beliefs
    recon_y_no_belief, m_no_belief, var_no_belief = model(x, y, adj, context)
    print(f"Forward pass without beliefs successful")
    
    # Test inference
    z = torch.randn(1, 1, args.cvae_hidden)
    pred_y = model.inference(x, z, adj, context, belief_sequences, belief_lengths)
    print(f"Inference successful")
    print(f"  Prediction shape: {pred_y[0].shape}")
    
    print("BeliefAwareTrajAirNet test passed!")


if __name__ == "__main__":
    test_belief_trajairnet()