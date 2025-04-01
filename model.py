import torch.nn as nn

from eigenmech import EigenmechanicsExtractor
from decoder import HierarchicalMotionDecoder
from physics import PhysicsConstraints

class PMAE(nn.Module):
    """
    Complete Proteomechanical Autoencoder model.
    """
    def __init__(self, input_dim, latent_dim, n_eigenmodes=100, 
                 use_forces=True, use_secondary_structure=True,
                 use_temperature=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_eigenmodes = n_eigenmodes
        self.use_forces = use_forces
        self.use_secondary_structure = use_secondary_structure
        self.use_temperature = use_temperature
        
        # Eigenmechanics extractor
        self.extractor = EigenmechanicsExtractor(
            input_dim, latent_dim, n_eigenmodes,
            use_forces, use_secondary_structure, use_temperature
        )
        
        # Motion decoder
        self.decoder = HierarchicalMotionDecoder(
            n_eigenmodes, latent_dim, input_dim, use_temperature
        )
        
        # Physics constraints module
        self.physics_constraints = PhysicsConstraints(input_dim)
    
    def forward(self, x, temperature=None, forces=None, dssp=None):
        """
        Forward pass through the PMAE model.
        
        Args:
            x: Protein trajectory data [batch_size, time_steps, input_dim]
            temperature: Temperature value [batch_size]
            forces: Force data [batch_size, time_steps, input_dim]
            dssp: Secondary structure data [batch_size, time_steps, ss_dim]
        """
        # Extract eigenmechanics
        mechanics = self.extractor(x, temperature, forces, dssp)
        
        # Reconstruct motion
        reconstruction = self.decoder(
            mechanics['eigenmodes'], 
            mechanics['mode_classes'], 
            temperature
        )
        
        # Predict forces if needed
        if self.use_forces:
            predicted_forces = self.decoder.predict_forces(reconstruction)
        else:
            predicted_forces = None
        
        return {
            'reconstruction': reconstruction,
            'eigenmodes': mechanics['eigenmodes'],
            'mode_classes': mechanics['mode_classes'],
            'latent': mechanics['latent'],
            'cath_pred': mechanics['cath_pred'],
            'predicted_forces': predicted_forces
        }