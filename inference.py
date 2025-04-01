import torch
import numpy as np

def predict_protein_motion(model, pdb_file, temperatures=(320, 348, 379, 413, 450), device='cuda'):
    """
    Predict protein motion for a new structure at different temperatures.
    
    Args:
        model: Trained PMAE model
        pdb_file: Path to PDB file
        temperatures: List of temperatures to simulate
        device: Device to run inference on
        
    Returns:
        Dictionary with predicted trajectories
    """
    import mdtraj as md
    
    # Load structure
    structure = md.load(pdb_file)
    
    # Extract coordinates
    coords = structure.xyz[0]  # Shape: [n_atoms, 3]
    
    # Convert to tensor
    coords_tensor = torch.tensor(coords.reshape(1, 1, -1), dtype=torch.float32).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        for temp in temperatures:
            # Convert temperature to tensor
            temp_tensor = torch.tensor([temp], dtype=torch.int64).to(device)
            
            # Forward pass to extract eigenmodes
            output = model(coords_tensor, temp_tensor)
            
            # Get eigenmodes
            eigenmodes = output['eigenmodes']
            mode_classes = output['mode_classes']
            
            # Generate trajectory by applying eigenmodes over time
            trajectory = []
            
            # Start with initial coordinates
            current_coords = coords_tensor.clone()
            trajectory.append(current_coords.cpu().numpy())
            
            # Generate 50 frames
            for t in range(50):
                # Scale eigenmodes over time (gradually increasing effect)
                time_scale = min(1.0, t / 25)
                
                # Apply scaled eigenmodes
                output = model.decoder(
                    eigenmodes * time_scale,
                    mode_classes,
                    temp_tensor
                )
                
                # Add to trajectory
                trajectory.append(output.cpu().numpy())
            
            # Convert to numpy array
            trajectory = np.concatenate(trajectory, axis=1)  # [1, 50, n_atoms*3]
            
            # Reshape to [50, n_atoms, 3]
            n_atoms = coords.shape[0]
            trajectory = trajectory.reshape(50, n_atoms, 3)
            
            # Store result
            results[temp] = {
                'trajectory': trajectory,
                'eigenmodes': eigenmodes.cpu().numpy(),
                'mode_classes': mode_classes.cpu().numpy()
            }
    
    return results

def analyze_protein_mechanics(model, data_loader, n_samples=100, device='cuda'):
    """
    Run inference on a dataset and extract key proteomechanics characteristics.
    
    Args:
        model: Trained PMAE model
        data_loader: DataLoader with protein structures
        n_samples: Number of samples to analyze
        device: Device to run inference on
        
    Returns:
        DataFrame with proteomechanical properties
    """
    import pandas as pd
    from tqdm import tqdm
    
    # Set model to evaluation mode
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Analyzing Protein Mechanics")):
            if i >= n_samples:
                break
                
            # Move data to device
            coords = batch['coords'].to(device)
            
            temperature = batch.get('temperature')
            if temperature is not None:
                temperature = temperature.to(device)
                
            forces = batch.get('forces')
            if forces is not None:
                forces = forces.to(device)
                
            dssp = batch.get('dssp_one_hot')
            if dssp is not None:
                dssp = dssp.to(device)
            
            # Forward pass
            output = model(coords, temperature, forces, dssp)
            
            # Extract eigenmodes
            eigenmodes = output['eigenmodes'].cpu().numpy()
            
            # Calculate eigenmode statistics
            for batch_idx in range(eigenmodes.shape[0]):
                domain_id = batch['domain_id'][batch_idx]
                temp = temperature[batch_idx].item() if temperature is not None else None
                
                # Get eigenmodes for this sample
                sample_modes = eigenmodes[batch_idx]
                
                # Calculate eigenmode statistics
                mode_mean = np.mean(sample_modes, axis=0)
                mode_std = np.std(sample_modes, axis=0)
                mode_max = np.max(np.abs(sample_modes), axis=0)
                
                # Calculate top 5 dominant modes (highest variance)
                mode_var = np.var(sample_modes, axis=0)
                top_modes = np.argsort(mode_var)[-5:][::-1]
                
                # Calculate motion characteristics
                if 'rmsd' in batch:
                    rmsd = batch['rmsd'][batch_idx].mean().item()
                else:
                    rmsd = None
                    
                if 'gyrationRadius' in batch:
                    gyration = batch['gyrationRadius'][batch_idx].mean().item()
                else:
                    gyration = None
                
                # Store results
                results.append({
                    'domain_id': domain_id,
                    'temperature': temp,
                    'top_modes': top_modes.tolist(),
                    'mode_mean': mode_mean,
                    'mode_std': mode_std,
                    'mode_max': mode_max,
                    'rmsd': rmsd,
                    'gyration_radius': gyration,
                })
    
    return pd.DataFrame(results)