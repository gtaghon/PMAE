import torch.nn as nn
from model import PMAE

def debug_tensor_shapes(model, batch):
    """Debug utility to print tensor shapes at various stages of the model."""
    print("\n=== DEBUG TENSOR SHAPES ===")
    print(f"Batch coords: {batch['coords'].shape}")
    
    if 'forces' in batch:
        print(f"Batch forces: {batch['forces'].shape}")
    
    if 'dssp_one_hot' in batch:
        print(f"Batch DSSP: {batch['dssp_one_hot'].shape}")
    
    # Test forward passes of components
    batch_size, time_steps, input_dim = batch['coords'].shape
    
    # Check encoder dimensions
    encoder_input_dim = model.extractor.encoder_layers[0].in_features
    print(f"Encoder expected input dim: {encoder_input_dim}")
    print(f"Available input dim: {input_dim}")
    
    # Calculate total dimensions
    total_dim = input_dim
    if 'forces' in batch and model.use_forces:
        total_dim += batch['forces'].shape[2]
    
    if 'dssp_one_hot' in batch and model.use_secondary_structure:
        if len(batch['dssp_one_hot'].shape) == 4:
            ss_dim = batch['dssp_one_hot'].shape[2] * batch['dssp_one_hot'].shape[3]
        else:
            ss_dim = batch['dssp_one_hot'].shape[2]
        total_dim += ss_dim
    
    print(f"Total calculated input dim: {total_dim}")
    print("===========================\n")

def initialize_and_check_model(model, sample_batch, device):
    """
    Initialize model parameters correctly based on the actual input dimensions.
    """
    print("\n=== Checking model dimensions ===")
    
    # Extract dimensions from sample batch
    coords_shape = sample_batch['coords'].shape
    batch_size, time_steps, input_dim = coords_shape
    print(f"Coordinate shape: {coords_shape}")
    
    # Calculate total input dimensions for the features after flattening
    coord_dim = input_dim
    total_feature_dim = coord_dim  # Start with coordinate dimension
    
    if 'forces' in sample_batch and model.use_forces:
        force_dim = sample_batch['forces'].shape[2]
        total_feature_dim += force_dim
        print(f"Adding forces dimension: {force_dim}")
    
    if 'dssp_one_hot' in sample_batch and model.use_secondary_structure:
        if len(sample_batch['dssp_one_hot'].shape) == 4:
            # If DSSP is [batch, time, residues, classes]
            num_residues = sample_batch['dssp_one_hot'].shape[2]
            ss_classes = sample_batch['dssp_one_hot'].shape[3]
            ss_dim = num_residues * ss_classes
            print(f"DSSP shape: {sample_batch['dssp_one_hot'].shape}")
            print(f"Flattened DSSP dimension: {ss_dim}")
        else:
            # If DSSP is already flattened per residue
            ss_dim = sample_batch['dssp_one_hot'].shape[2]
            print(f"Using provided DSSP dimension: {ss_dim}")
        
        total_feature_dim += ss_dim
    
    print(f"Total feature dimension calculated: {total_feature_dim}")
    
    # Create new model with correct dimensions
    new_model = PMAE(
        input_dim=input_dim,  # Original coordinate dimension 
        latent_dim=model.latent_dim,
        n_eigenmodes=model.n_eigenmodes,
        use_forces=model.use_forces,
        use_secondary_structure=model.use_secondary_structure,
        use_temperature=model.use_temperature
    )
    
    # Manually fix the encoder input layer
    if hasattr(new_model.extractor, 'encoder_layers') and len(new_model.extractor.encoder_layers) > 0:
        print(f"First layer input features before: {new_model.extractor.encoder_layers[0].in_features}")
        new_model.extractor.encoder_layers[0] = nn.Linear(total_feature_dim, 1024)
        print(f"First layer input features after: {new_model.extractor.encoder_layers[0].in_features}")
    else:
        print("WARNING: Could not find encoder_layers to adjust dimensions")
    
    return new_model.to(device)