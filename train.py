import torch
import torch.nn as nn
import torch.nn.functional as F

def train_pmae(model, data_loader, num_epochs=100, lr=0.001, device='cuda', 
               log_interval=10, save_dir='checkpoints'):
    """
    Train the PMAE model on mdCATH data.
    """
    import torch.optim as optim
    from torch.nn.utils import clip_grad_norm_
    import os
    from datetime import datetime
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    # Training loop
    global_step = 0
    start_time = datetime.now()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(data_loader):
            # Move data to device
            coords = batch['coords'].to(device)

            # NEW: Debug first batch of each epoch
            if batch_idx == 0 and epoch == 0:
                print(f"\n=== Debug Batch [{batch_idx}] of Epoch [{epoch}] ===")
                print(f"Coords shape: {coords.shape}")
                
                if 'forces' in batch:
                    print(f"Forces shape: {batch['forces'].shape}")
                
                if 'dssp_one_hot' in batch:
                    print(f"DSSP shape: {batch['dssp_one_hot'].shape}")
            
            # Optional inputs
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
            
            # Reconstruction loss
            recon_loss = mse_loss(output['reconstruction'], coords)
            
            # Force prediction loss (if applicable)
            force_loss = 0
            if forces is not None and output['predicted_forces'] is not None:
                force_loss = mse_loss(output['predicted_forces'], forces)
            
            # CATH class prediction loss (if applicable)
            cath_loss = 0
            if 'cath_class' in batch:
                cath_labels = batch['cath_class'].to(device)
                cath_pred = output['cath_pred'].reshape(-1, 4)
                cath_labels = cath_labels.reshape(-1)
                cath_loss = ce_loss(cath_pred, cath_labels)
            
            # Mode diversity loss - ensure eigenmodes are different
            batch_size, time_steps, n_modes = output['eigenmodes'].shape
            flat_modes = output['eigenmodes'].reshape(-1, n_modes)
            
            # Calculate correlation matrix (normalized dot products)
            normalized_modes = F.normalize(flat_modes, dim=0)
            mode_correlations = torch.mm(normalized_modes.T, normalized_modes)
            
            # Diversity loss: penalize high non-diagonal correlations
            identity = torch.eye(n_modes, device=device)
            diversity_loss = torch.mean(torch.abs(mode_correlations - identity))
            
            # Physics constraints loss
            physics_loss = 0
            if hasattr(model, 'physics_constraints'):
                # In practice, you would extract bond_indices and angle_indices from the protein topology
                # For now, pass None to calculate a placeholder physics loss
                physics_loss = model.physics_constraints(output['reconstruction'], None, None)
            
            # Total loss
            total_loss = (
                recon_loss + 
                0.1 * force_loss + 
                0.05 * cath_loss + 
                0.1 * diversity_loss + 
                0.1 * physics_loss
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            global_step += 1
            
            # Log progress
            if batch_idx % log_interval == 0:
                elapsed = datetime.now() - start_time
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(data_loader)} | "
                      f"Loss: {total_loss.item():.4f} | Recon: {recon_loss.item():.4f} | "
                      f"Force: {force_loss:.4f} | CATH: {cath_loss:.4f} | "
                      f"Diversity: {diversity_loss.item():.4f} | Physics: {physics_loss:.4f} | "
                      f"Elapsed: {elapsed}")
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_epoch_loss:.4f}")
        
        # Update learning rate
        scheduler.step(avg_epoch_loss)
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f"pmae_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")