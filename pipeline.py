import os
import torch
from torch.utils.data import DataLoader, random_split

from dataset import MdCATHDataset, collate_padded_batch
from model import PMAE
from eigenmech import EigenmechanicsCatalog
from inference import analyze_protein_mechanics
from train import train_pmae
from utils import initialize_and_check_model

# Add a hook to ensure the first forward pass works correctly
def hook_for_first_batch(data_loader, model):
    # Get the first batch
    first_batch = next(iter(data_loader))
    coords = first_batch['coords'].to(device)
    
    # Process DSSP to determine its actual shape
    dssp = first_batch.get('dssp_one_hot')
    if dssp is not None:
        dssp = dssp.to(device)
        
        # Calculate actual DSSP feature dimension
        if len(dssp.shape) == 4:
            num_residues = dssp.shape[2]
            ss_classes = dssp.shape[3]
            dssp_dim = num_residues * ss_classes
        else:
            dssp_dim = dssp.shape[2]
            
        print(f"DSSP actual dimension: {dssp_dim}")
    
    # Do a single forward pass with tracing to calculate dimensions
    with torch.no_grad():
        try:
            # This will initialize the networks correctly
            _ = model(coords, None, None, dssp)
            print("First forward pass for initialization successful")
        except Exception as e:
            print(f"Error in first forward pass (this is expected): {e}")

def run_pmae_pipeline(data_dir,
                      output_dir,
                      device,
                      batch_size=32,
                      num_epochs=100, 
                      latent_dim=128,
                      n_eigenmodes=64,
                      lr=0.001):
    """
    Run the complete PMAE training pipeline.
    
    Args:
        data_dir: Directory with mdCATH HDF5 files
        output_dir: Directory to save outputs
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        latent_dim: Dimension of latent space
        n_eigenmodes: Number of eigenmodes to extract
        lr: Learning rate
        
    Returns:
        Trained model and results
    """ 
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset
    dataset = MdCATHDataset(
        data_dir=data_dir,
        temps=(320, 348, 379, 413, 450),
        replicas=range(5),
        max_frames=100,
        frame_stride=4,
        include_forces=True,
        include_metadata=True
    )
    
    # Split into training and validation sets
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_padded_batch)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_padded_batch)
    
    # Get sample to determine input dimension
    sample = next(iter(train_loader))
    input_dim = sample['coords'].shape[2]
    
    # Initialize model
    model = PMAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_eigenmodes=n_eigenmodes,
        use_forces=True,
        use_secondary_structure=True,
        use_temperature=True
    )

    # NEW: Check and fix model dimensions before training
    model = initialize_and_check_model(model, sample, device)
    
    # Call the hook
    print("\n=== Performing initial forward pass for dynamic initialization ===")
    hook_for_first_batch(train_loader, model)
    
    # Train model
    train_pmae(
        model=model,
        data_loader=train_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        save_dir=os.path.join(output_dir, 'checkpoints')
    )
    
    # Create eigenmechanics catalog
    catalog = EigenmechanicsCatalog(model, val_loader)
    eigenmechanics_data = catalog.extract_eigenmechanics(num_samples=500)
    eigenmode_catalog = catalog.cluster_eigenmechanics(eigenmechanics_data, n_clusters=20)
    
    # Save eigenmechanics catalog
    torch.save(eigenmode_catalog, os.path.join(output_dir, 'eigenmode_catalog.pt'))
    
    # Analyze temperature dependence
    temp_distribution = catalog.analyze_temperature_dependence(eigenmode_catalog)
    torch.save(temp_distribution, os.path.join(output_dir, 'temperature_distribution.pt'))
    
    # Run inference on validation set
    mechanics_df = analyze_protein_mechanics(model, val_loader, n_samples=100)
    mechanics_df.to_csv(os.path.join(output_dir, 'protein_mechanics.csv'))
    
    return {
        'model': model,
        'eigenmode_catalog': eigenmode_catalog,
        'temp_distribution': temp_distribution,
        'mechanics_df': mechanics_df
    }

def get_accelerator():
    """
    Gets the fastest accelerator on the system.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using {device} for acceleration.")
    return device

if __name__ == "__main__":

    device = get_accelerator()

    run_pmae_pipeline(data_dir='mdcath_data/data',
                      output_dir='model_out',
                      device=device)