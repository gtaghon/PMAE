import torch
import torch.nn as nn

class HierarchicalMotionDecoder(nn.Module):
    """
    Reconstruct protein motion from eigenmechanics representation.
    """
    def __init__(self, n_eigenmodes, latent_dim, output_dim, use_temperature=True):
        super().__init__()
        self.n_eigenmodes = n_eigenmodes
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_temperature = use_temperature
        
        # Temperature embedding if needed
        if use_temperature:
            self.temp_embedding = nn.Embedding(450, 16)  # Temperature up to 450K
        
        # Mode combiner
        self.mode_combiner = nn.Sequential(
            nn.Linear(n_eigenmodes, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, output_dim)
        )
        
        # Force predictor - optional component for physics constraints
        self.force_predictor = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, eigenmodes, mode_classes, temperature=None):
        """
        Reconstruct protein coordinates from eigenmechanics.
        
        Args:
            eigenmodes: Eigenmode coefficients [batch_size, time_steps, n_eigenmodes]
            mode_classes: Mode class weights [batch_size, time_steps, n_eigenmodes]
            temperature: Temperature value [batch_size]
        """
        batch_size, time_steps, _ = eigenmodes.shape
        
        # Apply mode weights to eigenmodes
        weighted_modes = eigenmodes * torch.sigmoid(mode_classes)
        
        # Combine modes to latent representation
        latent = self.mode_combiner(weighted_modes.reshape(-1, self.n_eigenmodes))
        
        # Apply temperature conditioning if needed
        if self.use_temperature and temperature is not None:
            temp_embed = self.temp_embedding(temperature)
            temp_embed = temp_embed.unsqueeze(1).expand(-1, time_steps, -1)
            temp_embed = temp_embed.reshape(batch_size * time_steps, -1)
            
            # Adjust latent representation based on temperature
            latent = latent * (1.0 + 0.05 * torch.tanh(temp_embed))
        
        # Decode to coordinate space
        coords = self.decoder(latent)
        coords = coords.reshape(batch_size, time_steps, self.output_dim)
        
        return coords
    
    def predict_forces(self, coords):
        """
        Predict forces from coordinates.
        """
        batch_size, time_steps, _ = coords.shape
        coords_flat = coords.reshape(-1, self.output_dim)
        forces = self.force_predictor(coords_flat)
        return forces.reshape(batch_size, time_steps, self.output_dim)