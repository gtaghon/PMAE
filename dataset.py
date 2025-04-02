import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tempfile
import os
from pathlib import Path

class MdCATHDataset(Dataset):
    """
    Dataset class for loading mdCATH data with consistent tensor sizes.
    """
    def __init__(self, data_dir, domain_ids=None, temps=(320, 348, 379, 413, 450), 
                 replicas=range(5), max_frames=100, frame_stride=1,
                 include_forces=True, include_metadata=True,
                 pad_or_truncate=True, max_atoms=None):
        """
        Initialize the mdCATH dataset with consistent tensor sizing.
        """
        self.data_dir = Path(data_dir)
        self.temps = [str(t) for t in temps]
        self.replicas = [str(r) for r in replicas]
        self.max_frames = max_frames
        self.frame_stride = frame_stride
        self.include_forces = include_forces
        self.include_metadata = include_metadata
        self.pad_or_truncate = pad_or_truncate
        self.max_atoms = max_atoms
        
        # Get all domain files if domain_ids is None
        if domain_ids is None:
            self.domain_files = list(self.data_dir.glob("*.h5"))
        else:
            self.domain_files = [self.data_dir / f for f in domain_ids if Path(self.data_dir / f).exists()]
            
        self.trajectories = []
        
        # Index all trajectories and determine max_atoms if not provided
        atom_counts = []
        
        for domain_file in self.domain_files:
            try:
                with h5py.File(domain_file, 'r') as f:
                    domain_id = next(iter(f.keys()))  # Get the domain ID from the file
                    
                    # Count atoms
                    num_atoms = len(f[domain_id]['chain'])
                    atom_counts.append(num_atoms)
                    
                    for temp in self.temps:
                        if temp not in f[domain_id]:
                            continue
                            
                        for replica in self.replicas:
                            if replica not in f[domain_id][temp]:
                                continue
                                
                            # Get trajectory info
                            num_frames = f[domain_id][temp][replica].attrs.get('numFrames', 0)
                            
                            if num_frames > 0:
                                self.trajectories.append({
                                    'domain_id': domain_id,
                                    'domain_file': domain_file,
                                    'temp': temp,
                                    'replica': replica,
                                    'num_frames': num_frames,
                                    'num_atoms': num_atoms
                                })
            except Exception as e:
                print(f"Error loading {domain_file}: {e}")
        
        # Determine max_atoms if not provided
        if self.max_atoms is None and self.pad_or_truncate:
            if atom_counts:
                # Use median instead of 95th percentile to avoid extremely large proteins
                self.max_atoms = int(np.median(atom_counts))
                print(f"Auto-determined max_atoms: {self.max_atoms}")
            else:
                self.max_atoms = 500  # Default fallback
                print(f"Using default max_atoms: {self.max_atoms}")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """
        Get a trajectory item with padded/truncated data for consistent sizes.
        """
        traj_info = self.trajectories[idx]
        domain_id = traj_info['domain_id']
        domain_file = traj_info['domain_file']
        temp = traj_info['temp']
        replica = traj_info['replica']
        
        try:
            with h5py.File(domain_file, 'r') as f:
                # Determine frames to load (with truncation)
                frames = min(traj_info['num_frames'], self.max_frames if self.max_frames else float('inf'))
                selected_frames = list(range(0, frames, self.frame_stride))
                actual_frames = len(selected_frames)
                
                # Load coordinates (convert from Å to nm)
                coords = f[domain_id][temp][replica]['coords'][selected_frames]
                coords = coords / 10.0  # Convert from Å to nm
                
                # IMPORTANT FIX: Handle proper reshaping for atom coordinates
                # First check if coords are already in the shape [frames, atoms, 3]
                if len(coords.shape) == 3 and coords.shape[2] == 3:
                    # Already in the correct format
                    pass
                elif len(coords.shape) == 2:
                    # Reshape from [frames, atoms*3] to [frames, atoms, 3]
                    num_atoms = coords.shape[1] // 3
                    coords = coords.reshape(coords.shape[0], num_atoms, 3)
                
                # Now handle padding/truncation for atoms with proper 3D structure
                num_atoms = traj_info['num_atoms']
                if self.pad_or_truncate and self.max_atoms:
                    if num_atoms > self.max_atoms:
                        # Truncate to max_atoms
                        coords = coords[:, :self.max_atoms, :]
                    elif num_atoms < self.max_atoms:
                        # Pad with zeros - create a new array with the right shape
                        padded_coords = np.zeros((coords.shape[0], self.max_atoms, 3), dtype=coords.dtype)
                        padded_coords[:, :num_atoms, :] = coords
                        coords = padded_coords
                
                # Finally, reshape to the expected format for the model: [frames, atoms*3]
                coords = coords.reshape(coords.shape[0], -1)
                
                # Load forces if requested
                forces = None
                if self.include_forces:
                    try:
                        forces = f[domain_id][temp][replica]['forces'][selected_frames]
                        # Handle padding/truncation for forces
                        if self.pad_or_truncate and self.max_atoms:
                            if num_atoms*3 > self.max_atoms*3:
                                forces = forces[:, :self.max_atoms*3]
                            elif num_atoms*3 < self.max_atoms*3:
                                # Pad with zeros
                                padded_forces = np.zeros((forces.shape[0], self.max_atoms*3), dtype=forces.dtype)
                                padded_forces[:, :forces.shape[1]] = forces
                                forces = padded_forces
                    except (KeyError, ValueError, IndexError) as e:
                        # Silently fail if forces aren't available
                        forces = None
                
                # Load secondary structure if available and requested
                dssp = None
                if self.include_metadata and 'dssp' in f[domain_id][temp][replica]:
                    try:
                        dssp = f[domain_id][temp][replica]['dssp'][selected_frames]
                        
                        # Calculate number of residues
                        n_residues = dssp.shape[1]
                        
                        # Handle padding/truncation for DSSP
                        if self.pad_or_truncate and self.max_atoms:
                            # Estimate max residues from max_atoms (about 1 residue per 10 atoms)
                            max_residues = max(1, self.max_atoms // 10)
                            
                            if n_residues > max_residues:
                                # Truncate to max_residues
                                dssp = dssp[:, :max_residues]
                            elif n_residues < max_residues:
                                # Pad with empty DSSP code (loop/coil)
                                padded_dssp = np.empty((dssp.shape[0], max_residues), dtype=dssp.dtype)
                                padded_dssp.fill(b' ')  # Space character for coil
                                padded_dssp[:, :n_residues] = dssp
                                dssp = padded_dssp
                    except Exception as e:
                        # Silently fail if DSSP isn't available
                        dssp = None
                
                # Load other metadata
                metadata = {}
                if self.include_metadata:
                    for field in ['rmsd', 'gyrationRadius']:
                        if field in f[domain_id][temp][replica]:
                            try:
                                metadata[field] = f[domain_id][temp][replica][field][selected_frames]
                            except Exception:
                                pass
                    
                    # RMSF is per residue, not per frame
                    if 'rmsf' in f[domain_id][temp][replica]:
                        try:
                            rmsf = f[domain_id][temp][replica]['rmsf'][:]
                            
                            # Handle padding/truncation for RMSF
                            if self.pad_or_truncate and self.max_atoms:
                                # Estimate max residues from max_atoms
                                max_residues = max(1, self.max_atoms // 10)
                                
                                if len(rmsf) > max_residues:
                                    rmsf = rmsf[:max_residues]
                                elif len(rmsf) < max_residues:
                                    padded_rmsf = np.zeros(max_residues, dtype=rmsf.dtype)
                                    padded_rmsf[:len(rmsf)] = rmsf
                                    rmsf = padded_rmsf
                            
                            metadata['rmsf'] = rmsf
                        except Exception:
                            pass
                
                # Get atom metadata (limited to max_atoms)
                atom_data = {}
                for field in ['resid', 'resname', 'chain', 'element']:
                    if field in f[domain_id]:
                        try:
                            data = f[domain_id][field][:]
                            
                            # Handle padding/truncation for atom metadata
                            if self.pad_or_truncate and self.max_atoms:
                                if len(data) > self.max_atoms:
                                    data = data[:self.max_atoms]
                                elif len(data) < self.max_atoms:
                                    # Create a padded array with the appropriate dtype
                                    pad_value = b'' if data.dtype.kind == 'S' else 0
                                    padded_data = np.empty(self.max_atoms, dtype=data.dtype)
                                    
                                    # Fill with appropriate padding value
                                    if data.dtype.kind == 'S':
                                        padded_data.fill(b'')
                                    else:
                                        padded_data.fill(0)
                                    
                                    # Copy the original data
                                    padded_data[:len(data)] = data
                                    data = padded_data
                            
                            atom_data[field] = data
                        except Exception:
                            pass
                
                # Process secondary structure (convert to one-hot)
                if dssp is not None:
                    try:
                        dssp_one_hot = self._process_dssp(dssp)
                    except Exception:
                        dssp_one_hot = None
                else:
                    dssp_one_hot = None
                
                # Convert to tensors
                coords_tensor = torch.tensor(coords, dtype=torch.float32)
                
                result = {
                    'domain_id': domain_id,
                    'temperature': int(temp),
                    'replica': int(replica),
                    'coords': coords_tensor,
                    'atom_data': atom_data,
                    'original_num_atoms': num_atoms,
                }
                
                if forces is not None:
                    result['forces'] = torch.tensor(forces, dtype=torch.float32)
                    
                if dssp_one_hot is not None:
                    result['dssp_one_hot'] = torch.tensor(dssp_one_hot, dtype=torch.float32)
                
                for key, value in metadata.items():
                    if value is not None:
                        result[key] = torch.tensor(value, dtype=torch.float32)
                
                return result
                
        except Exception as e:
            print(f"Error processing {domain_file}, {domain_id}, {temp}, {replica}: {e}")
            # Return a minimal valid item that won't crash the loader
            # This is a fallback for corrupted data
            min_frames = 10 if self.max_frames is None else min(10, self.max_frames)
            atom_dim = 300 if self.max_atoms is None else self.max_atoms * 3
            
            return {
                'domain_id': domain_id,
                'temperature': int(temp) if temp.isdigit() else 320,
                'replica': int(replica) if replica.isdigit() else 0,
                'coords': torch.zeros((min_frames, atom_dim), dtype=torch.float32),
                'atom_data': {},
                'original_num_atoms': 0,
            }
    
    def _process_dssp(self, dssp):
        """
        Process DSSP data into one-hot encoding.
        """
        # 8-class DSSP encoding: H, G, I, E, B, T, S, and loop (blank)
        dssp_codes = "HGIEBTS "
        
        # One-hot encoded tensor
        one_hot = np.zeros((dssp.shape[0], dssp.shape[1], len(dssp_codes)), dtype=np.float32)
        
        # Fill the tensor
        for i in range(dssp.shape[0]):
            for j in range(dssp.shape[1]):
                try:
                    code = dssp[i, j].decode()
                    if code in dssp_codes:
                        idx = dssp_codes.index(code)
                        one_hot[i, j, idx] = 1.0
                    else:
                        # Default to coil (space) if not in codes
                        one_hot[i, j, 7] = 1.0
                except:
                    # Default to coil (space) if decoding fails
                    one_hot[i, j, 7] = 1.0
        
        return one_hot
    
    def get_solid_fraction(self, dssp):
        """
        Compute the solid fraction (α+β structure) from DSSP data.
        Adapted from mdCATH analysis/utils.py
        """
        # Maps for secondary structure classification
        float_map = {"H": 0, "B": 1, "E": 1, "G": 0, "I": 0, "T": 2, "S": 2, " ": 2}  # 3-type differentiation
        
        dssp_decoded_float = np.zeros((dssp.shape[0], dssp.shape[1]), dtype=np.float32)
        for i in range(dssp.shape[0]):
            dssp_decoded_float[i] = [float_map[el.decode()] for el in dssp[i]]
        
        # Calculate fraction of alpha (0) and beta (1) structures
        solid_fraction = np.logical_or(dssp_decoded_float == 0, dssp_decoded_float == 1)
        
        return solid_fraction

def collate_padded_batch(batch):
    """
    Custom collate function to handle batches with different sizes.
    """
    # Filter out problematic items
    valid_batch = [item for item in batch if isinstance(item, dict) and 'coords' in item and len(item['coords'].shape) == 2]
    
    if not valid_batch:
        raise ValueError("No valid items in batch")
    
    # Get maximum sequence length in this batch
    max_frames = max([item['coords'].shape[0] for item in valid_batch])
    
    # Batch items
    batch_dict = {
        'domain_id': [],
        'temperature': [],
        'replica': [],
        'coords': [],
        'original_num_atoms': [],
    }
    
    # Optional fields to include if available
    optional_fields = ['forces', 'dssp_one_hot', 'rmsd', 'gyrationRadius', 'rmsf']
    optional_exists = {field: all(field in item for item in valid_batch) for field in optional_fields}
    
    for field in optional_fields:
        if optional_exists[field]:
            batch_dict[field] = []
    
    # Process each item
    for item in valid_batch:
        # Add simple fields
        batch_dict['domain_id'].append(item['domain_id'])
        batch_dict['temperature'].append(item['temperature'])
        batch_dict['replica'].append(item['replica'])
        batch_dict['original_num_atoms'].append(item.get('original_num_atoms', 0))
        
        # Handle padded tensors
        frames = item['coords'].shape[0]
        
        # Pad coordinates if needed
        if frames < max_frames:
            padding = torch.zeros((max_frames - frames, item['coords'].shape[1]), dtype=item['coords'].dtype)
            padded_coords = torch.cat([item['coords'], padding], dim=0)
        else:
            padded_coords = item['coords']
        
        batch_dict['coords'].append(padded_coords)
        
        # Pad optional tensors
        for field in optional_fields:
            if optional_exists[field] and field in item:
                tensor = item[field]
                
                # Handle different tensor shapes
                if len(tensor.shape) == 1:  # 1D tensor (e.g., rmsf)
                    batch_dict[field].append(tensor)
                elif len(tensor.shape) == 2:  # 2D tensor (e.g., rmsd per frame)
                    if tensor.shape[0] < max_frames:
                        padding = torch.zeros((max_frames - tensor.shape[0], tensor.shape[1]), 
                                              dtype=tensor.dtype)
                        padded_tensor = torch.cat([tensor, padding], dim=0)
                    else:
                        padded_tensor = tensor
                    batch_dict[field].append(padded_tensor)
                elif len(tensor.shape) == 3:  # 3D tensor (e.g., dssp_one_hot)
                    if tensor.shape[0] < max_frames:
                        padding = torch.zeros((max_frames - tensor.shape[0], tensor.shape[1], tensor.shape[2]), 
                                              dtype=tensor.dtype)
                        padded_tensor = torch.cat([tensor, padding], dim=0)
                    else:
                        padded_tensor = tensor
                    batch_dict[field].append(padded_tensor)
    
    # Convert lists to tensors where appropriate
    batch_dict['temperature'] = torch.tensor(batch_dict['temperature'])
    batch_dict['replica'] = torch.tensor(batch_dict['replica'])
    batch_dict['original_num_atoms'] = torch.tensor(batch_dict['original_num_atoms'])
    batch_dict['coords'] = torch.stack(batch_dict['coords'])
    
    # Process optional fields
    for field in optional_fields:
        if optional_exists[field] and batch_dict[field]:
            # Check if all tensors have the same shape (for stacking)
            shapes = [tensor.shape for tensor in batch_dict[field]]
            if all(shape == shapes[0] for shape in shapes):
                batch_dict[field] = torch.stack(batch_dict[field])
    
    # Handle atom_data separately
    atom_data = {}
    
    return batch_dict