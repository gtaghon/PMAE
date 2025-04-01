import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsConstraints(nn.Module):
    """
    Apply physics-based constraints to ensure plausible protein motion.
    """
    def __init__(self, coord_dim):
        super().__init__()
        self.coord_dim = coord_dim
    
    def bond_length_constraints(self, coords, bond_indices):
        """
        Apply bond length constraints using harmonic potential.
        
        Args:
            coords: Protein coordinates [batch_size, time_steps, coord_dim]
            bond_indices: Indices of atom pairs forming bonds [num_bonds, 2]
        """
        batch_size, time_steps, _ = coords.shape
        
        # Reshape for easier indexing
        coords_flat = coords.reshape(batch_size * time_steps, -1, 3)
        
        # Get coordinates of bonded atoms
        atom1 = coords_flat[:, bond_indices[:, 0]]
        atom2 = coords_flat[:, bond_indices[:, 1]]
        
        # Calculate bond lengths
        bond_vectors = atom1 - atom2
        bond_lengths = torch.norm(bond_vectors, dim=2)
        
        # Calculate harmonic energy (no target length for now, just consistency)
        energy = torch.var(bond_lengths, dim=1).mean()
        
        return energy
    
    def angle_constraints(self, coords, angle_indices):
        """
        Apply angle constraints using harmonic potential.
        
        Args:
            coords: Protein coordinates [batch_size, time_steps, coord_dim]
            angle_indices: Indices of atom triplets forming angles [num_angles, 3]
        """
        batch_size, time_steps, _ = coords.shape
        
        # Reshape for easier indexing
        coords_flat = coords.reshape(batch_size * time_steps, -1, 3)
        
        # Get coordinates of atoms forming angles
        atom1 = coords_flat[:, angle_indices[:, 0]]
        atom2 = coords_flat[:, angle_indices[:, 1]]
        atom3 = coords_flat[:, angle_indices[:, 2]]
        
        # Calculate vectors
        v1 = atom1 - atom2
        v2 = atom3 - atom2
        
        # Normalize vectors
        v1_norm = F.normalize(v1, dim=2)
        v2_norm = F.normalize(v2, dim=2)
        
        # Calculate angles (dot product of normalized vectors)
        cos_angles = torch.sum(v1_norm * v2_norm, dim=2)
        
        # Constrain to valid range [-1, 1] and calculate variance
        cos_angles = torch.clamp(cos_angles, -1, 1)
        energy = torch.var(cos_angles, dim=1).mean()
        
        return energy
    
    def forward(self, coords, bond_indices=None, angle_indices=None):
        """
        Calculate physics-based constraint energies.
        
        Args:
            coords: Protein coordinates [batch_size, time_steps, coord_dim]
            bond_indices: Indices of atom pairs forming bonds [num_bonds, 2]
            angle_indices: Indices of atom triplets forming angles [num_angles, 3]
        """
        # For simplicity, if indices are not provided, return zero energy
        # In practice, these would come from the protein topology
        bond_energy = self.bond_length_constraints(coords, bond_indices) if bond_indices is not None else 0
        angle_energy = self.angle_constraints(coords, angle_indices) if angle_indices is not None else 0
        
        # Total constraint energy
        total_energy = bond_energy + angle_energy
        
        return total_energy