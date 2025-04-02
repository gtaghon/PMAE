import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EigenmechanicsExtractor(nn.Module):
    """
    Extract fundamental motion patterns (eigenmechanics) from protein trajectories.
    """
    def __init__(self, input_dim, latent_dim, n_eigenmodes=100, 
                use_forces=True, use_secondary_structure=True, 
                use_temperature=True):
        super().__init__()
        self.input_dim = input_dim  # Total flattened coordinate dimension
        self.latent_dim = latent_dim
        self.n_eigenmodes = n_eigenmodes
        self.use_forces = use_forces
        self.use_secondary_structure = use_secondary_structure
        self.use_temperature = use_temperature
        
        # Temperature embedding - handle all possible temperatures
        if use_temperature:
            self.temp_embedding = nn.Embedding(500, 16)  # Temperature up to 500K
        
        # PLACEHOLDER - the actual encoder_input_dim will be determined at runtime
        # This will be overridden when we process the first batch
        self.encoder_input_dim = input_dim  # Start with just coordinates
        
        # NOTE: We will NOT create encoder layers here, but in a separate initialize method
        # that will be called after seeing the actual data dimensions
        
        # Define encoder layer sizes for later instantiation
        self.encoder_layer_sizes = [1024, 512, latent_dim]
        
        # Initialize a minimal encoder to avoid errors if initialize() isn't called
        self.encoder_layers = nn.ModuleList([
            nn.Linear(input_dim, self.encoder_layer_sizes[0]),
            nn.LayerNorm(self.encoder_layer_sizes[0]),
            nn.SiLU(),
            nn.Linear(self.encoder_layer_sizes[0], self.encoder_layer_sizes[1]),
            nn.LayerNorm(self.encoder_layer_sizes[1]),
            nn.SiLU(),
            nn.Linear(self.encoder_layer_sizes[1], latent_dim)
        ])
        
        # Eigenmode extraction
        self.eigenmode_layers = nn.ModuleList([
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, n_eigenmodes)
        ])
        
        # Mode classifier
        self.classifier_layers = nn.ModuleList([
            nn.Linear(n_eigenmodes, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, n_eigenmodes)
        ])
        
        # CATH classification (4 main classes)
        self.cath_layers = nn.ModuleList([
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 4)
        ])

    def initialize_encoder(self, actual_input_dim):
        """
        Initialize the encoder with the correct input dimension, based on actual data.
        Should be called after seeing a sample of the data.
        """
        print(f"Initializing encoder with input dimension: {actual_input_dim}")
        self.encoder_input_dim = actual_input_dim
        
        # Create encoder layers with correct dimensions
        self.encoder_layers = nn.ModuleList([
            nn.Linear(actual_input_dim, self.encoder_layer_sizes[0]),
            nn.LayerNorm(self.encoder_layer_sizes[0]),
            nn.SiLU(),
            nn.Linear(self.encoder_layer_sizes[0], self.encoder_layer_sizes[1]),
            nn.LayerNorm(self.encoder_layer_sizes[1]),
            nn.SiLU(),
            nn.Linear(self.encoder_layer_sizes[1], self.latent_dim)
        ])
        
        print(f"Encoder initialized with layers: {[layer.in_features for layer in self.encoder_layers if isinstance(layer, nn.Linear)]}")
        return self
    
    def forward(self, x, temperature=None, forces=None, dssp=None):
        """
        Extract eigenmechanics features from trajectory data.
        
        Args:
            x: Protein trajectory data [batch_size, time_steps, input_dim]
            temperature: Temperature value [batch_size]
            forces: Force data [batch_size, time_steps, input_dim]
            dssp: Secondary structure data [batch_size, time_steps, num_residues, ss_dim]
        """
        batch_size, time_steps, _ = x.shape
        
        # DEBUG: Print shape information
        print(f"Input shape: {x.shape}")
        
        # Reshape for encoder
        x_flat = x.reshape(batch_size * time_steps, -1)
        
        # Construct input features
        features = [x_flat]
        
        # Add forces if provided and configured
        if self.use_forces and forces is not None:
            forces_flat = forces.reshape(batch_size * time_steps, -1)
            features.append(forces_flat)
                
        # Add secondary structure if provided and configured
        if self.use_secondary_structure and dssp is not None:
            try:
                # If dssp is [batch_size, time_steps, num_residues, ss_dim]
                if len(dssp.shape) == 4:
                    # For proper handling, flatten the residue and ss dimensions
                    batch_size_ds, time_steps_ds, num_residues, ss_dim = dssp.shape
                    dssp_flat = dssp.reshape(batch_size_ds * time_steps_ds, num_residues * ss_dim)
                # If dssp is already [batch_size, time_steps, flattened_dim]
                elif len(dssp.shape) == 3:
                    dssp_flat = dssp.reshape(batch_size * time_steps, -1)
                else:
                    dssp_flat = None
                    
                if dssp_flat is not None:
                    features.append(dssp_flat)
                    print(f"DSSP flattened shape: {dssp_flat.shape}")
            except Exception as e:
                print(f"Error processing DSSP in forward pass: {e}")
        
        # Concatenate all features
        try:
            combined_features = torch.cat(features, dim=1)
            print(f"Combined features shape: {combined_features.shape}")
            
            # IMPORTANT: Ensure the first layer can handle this dimension
            if combined_features.shape[1] != self.encoder_layers[0].in_features:
                print(f"WARNING: Input dimension mismatch - got {combined_features.shape[1]}, expected {self.encoder_layers[0].in_features}")
                # Dynamic adjustment of the first layer - risky but can work during debugging
                self.encoder_layers[0] = nn.Linear(combined_features.shape[1], 1024).to(combined_features.device)
                print(f"Dynamically adjusted first layer to accept {combined_features.shape[1]} features")
                
        except Exception as e:
            print(f"Error concatenating features: {e}")
            # Fallback to just coordinates
            combined_features = x_flat
        
        # Process through encoder
        latent_repr = combined_features
        for layer in self.encoder_layers:
            latent_repr = layer(latent_repr)
        
        # Apply temperature conditioning if provided
        if self.use_temperature and temperature is not None:
            try:
                # Clip temperatures to valid range
                clipped_temps = torch.clamp(temperature, 0, 499)
                temp_embed = self.temp_embedding(clipped_temps)
                
                # Broadcast temperature embedding
                temp_embed = temp_embed.unsqueeze(1).expand(-1, time_steps, -1)
                temp_embed = temp_embed.reshape(batch_size * time_steps, -1)
                
                # Condition latent representation
                latent_repr = latent_repr * (1.0 + 0.1 * torch.tanh(temp_embed))
            except Exception as e:
                print(f"Error in temperature conditioning: {e}")
        
        # Extract eigenmodes
        eigenmodes = latent_repr
        for layer in self.eigenmode_layers:
            eigenmodes = layer(eigenmodes)
        
        # Classify modes
        mode_classes = eigenmodes
        for layer in self.classifier_layers:
            mode_classes = layer(mode_classes)
        
        # CATH classification prediction
        cath_pred = latent_repr
        for layer in self.cath_layers:
            cath_pred = layer(cath_pred)
        
        # Reshape outputs back to batch and time dimensions
        latent_repr = latent_repr.reshape(batch_size, time_steps, self.latent_dim)
        eigenmodes = eigenmodes.reshape(batch_size, time_steps, self.n_eigenmodes)
        mode_classes = mode_classes.reshape(batch_size, time_steps, self.n_eigenmodes)
        cath_pred = cath_pred.reshape(batch_size, time_steps, 4)
        
        return {
            'latent': latent_repr,
            'eigenmodes': eigenmodes,
            'mode_classes': mode_classes,
            'cath_pred': cath_pred
        }

class EigenmechanicsCatalog:
    """
    Analyze and catalog eigenmechanics from trained PMAE models.
    """
    def __init__(self, model, data_loader, device='cuda'):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.model.eval()
        
    def extract_eigenmechanics(self, num_samples=1000, display_progress=True):
        """
        Extract eigenmechanics from protein trajectories.
        
        Args:
            num_samples: Maximum number of trajectories to analyze
            display_progress: Whether to display progress bar
        
        Returns:
            Dictionary containing eigenmechanics data
        """
        from tqdm import tqdm
        
        eigenmodes_collection = []
        mode_classes_collection = []
        domain_ids = []
        temperatures = []
        
        loader = tqdm(self.data_loader) if display_progress else self.data_loader
        count = 0
        
        with torch.no_grad():
            for batch in loader:
                if count >= num_samples:
                    break
                
                # Move data to device
                coords = batch['coords'].to(self.device)
                
                temperature = batch.get('temperature')
                if temperature is not None:
                    temperature = temperature.to(self.device)
                    
                forces = batch.get('forces')
                if forces is not None:
                    forces = forces.to(self.device)
                    
                dssp = batch.get('dssp_one_hot')
                if dssp is not None:
                    dssp = dssp.to(self.device)
                
                # Extract eigenmechanics
                output = self.model(coords, temperature, forces, dssp)
                
                # Store results
                eigenmodes_collection.append(output['eigenmodes'].cpu().numpy())
                mode_classes_collection.append(output['mode_classes'].cpu().numpy())
                domain_ids.extend(batch['domain_id'])
                
                if temperature is not None:
                    temperatures.extend(temperature.cpu().numpy())
                else:
                    temperatures.extend([None] * len(batch['domain_id']))
                
                count += len(batch['domain_id'])
                
                if display_progress:
                    loader.set_description(f"Extracted {count}/{min(num_samples, len(self.data_loader.dataset))} samples")
        
        return {
            'eigenmodes': np.concatenate(eigenmodes_collection, axis=0),
            'mode_classes': np.concatenate(mode_classes_collection, axis=0),
            'domain_ids': domain_ids,
            'temperatures': temperatures
        }
    
    def cluster_eigenmechanics(self, eigenmechanics_data, n_clusters=50, method='kmeans'):
        """
        Cluster eigenmechanics to identify common patterns.
        
        Args:
            eigenmechanics_data: Data from extract_eigenmechanics()
            n_clusters: Number of clusters for eigenmechanics
            method: Clustering method ('kmeans' or 'hierarchical')
            
        Returns:
            Dictionary containing clustered eigenmechanics
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Reshape eigenmodes for clustering
        eigenmodes = eigenmechanics_data['eigenmodes']
        batch_size, time_steps, n_modes = eigenmodes.shape
        flat_eigenmodes = eigenmodes.reshape(-1, n_modes)
        
        # Standardize data for better clustering
        scaler = StandardScaler()
        flat_eigenmodes_scaled = scaler.fit_transform(flat_eigenmodes)
        
        # Optional: reduce dimensions for faster clustering
        if n_modes > 50:
            pca = PCA(n_components=min(50, n_modes - 1))
            flat_eigenmodes_scaled = pca.fit_transform(flat_eigenmodes_scaled)
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
            
        cluster_labels = clusterer.fit_predict(flat_eigenmodes_scaled)
        
        # Create catalog of eigenmode clusters
        eigenmode_catalog = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            
            # Get samples from this cluster
            cluster_samples = flat_eigenmodes[cluster_mask][:100]  # Up to 100 examples
            
            # Get metadata for these samples
            flat_idx = np.where(cluster_mask)[0]
            batch_idx = flat_idx // time_steps
            time_idx = flat_idx % time_steps
            
            cluster_domains = [eigenmechanics_data['domain_ids'][idx] for idx in np.clip(batch_idx, 0, len(eigenmechanics_data['domain_ids'])-1)]
            
            if eigenmechanics_data['temperatures'][0] is not None:
                cluster_temps = [eigenmechanics_data['temperatures'][idx] for idx in np.clip(batch_idx, 0, len(eigenmechanics_data['temperatures'])-1)]
            else:
                cluster_temps = None
            
            # Store centroid and examples
            if method == 'kmeans':
                # For KMeans, we can get the actual centroid
                centroid = clusterer.cluster_centers_[i]
                if n_modes > 50:  # If we used PCA, transform back to original space
                    centroid = scaler.inverse_transform(pca.inverse_transform(centroid))
            else:
                # For hierarchical, just use the mean of samples
                centroid = np.mean(cluster_samples, axis=0)
            
            eigenmode_catalog[f"eigenmode_cluster_{i}"] = {
                "centroid": centroid,
                "samples": cluster_samples,
                "domains": cluster_domains[:100],
                "temperatures": cluster_temps[:100] if cluster_temps else None,
                "time_indices": time_idx[:100].tolist(),
                "sample_count": int(np.sum(cluster_mask))
            }
        
        return eigenmode_catalog
    
    def analyze_temperature_dependence(self, eigenmode_catalog):
        """
        Analyze how eigenmechanics change with temperature.
        
        Args:
            eigenmode_catalog: Output from cluster_eigenmechanics()
            
        Returns:
            Dictionary with temperature distribution per cluster
        """
        temp_distribution = {}
        
        for cluster_id, cluster_data in eigenmode_catalog.items():
            if cluster_data['temperatures'] is None:
                continue
                
            temps = np.array(cluster_data['temperatures'])
            
            # Count occurrences of each temperature
            unique_temps, temp_counts = np.unique(temps, return_counts=True)
            
            # Normalize to get distribution
            total = np.sum(temp_counts)
            distribution = {int(temp): count/total for temp, count in zip(unique_temps, temp_counts)}
            
            temp_distribution[cluster_id] = distribution
        
        return temp_distribution
    
    def analyze_cath_class_distribution(self, eigenmode_catalog, domain_to_cath_map):
        """
        Analyze CATH class distribution within each eigenmode cluster.
        
        Args:
            eigenmode_catalog: Output from cluster_eigenmechanics()
            domain_to_cath_map: Dictionary mapping domain IDs to CATH classes
            
        Returns:
            Dictionary with CATH class distribution per cluster
        """
        cath_distribution = {}
        
        for cluster_id, cluster_data in eigenmode_catalog.items():
            domains = cluster_data['domains']
            cath_counts = {'mainly_alpha': 0, 'mainly_beta': 0, 'alpha_beta': 0, 'few_ss': 0}
            
            for domain in domains:
                if domain in domain_to_cath_map:
                    cath_class = domain_to_cath_map[domain]
                    cath_counts[cath_class] = cath_counts.get(cath_class, 0) + 1
                    
            total = sum(cath_counts.values())
            if total > 0:
                cath_dist = {cath: count/total for cath, count in cath_counts.items()}
                cath_distribution[cluster_id] = cath_dist
        
        return cath_distribution
    
    def visualize_eigenmechanics(self, eigenmode_catalog, domain_loader, top_clusters=5):
        """
        Visualize the effect of applying eigenmechanics to protein structures.
        
        Args:
            eigenmode_catalog: Output from cluster_eigenmechanics()
            domain_loader: DataLoader with domain structures
            top_clusters: Number of top clusters to visualize
            
        Returns:
            Dictionary with visualization data
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Sort clusters by sample count
        sorted_clusters = sorted(
            eigenmode_catalog.items(), 
            key=lambda x: x[1]['sample_count'], 
            reverse=True
        )
        
        # Get representative domains for top clusters
        visualizations = {}
        
        for i, (cluster_id, cluster_data) in enumerate(sorted_clusters[:top_clusters]):
            # Get a representative domain
            rep_domain = cluster_data['domains'][0]
            rep_temp = cluster_data['temperatures'][0] if cluster_data['temperatures'] else None
            
            # Find this domain in the data loader
            rep_coords = None
            for batch in domain_loader:
                domain_indices = [i for i, d in enumerate(batch['domain_id']) if d == rep_domain]
                if domain_indices:
                    rep_coords = batch['coords'][domain_indices[0]].cpu().numpy()
                    break
            
            if rep_coords is None:
                continue
            
            # Get centroid eigenmode
            centroid = cluster_data['centroid']
            
            # Create a new figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot original structure (CA atoms only for clarity)
            ca_indices = np.arange(0, rep_coords.shape[1], 3)  # Approximate CA indices
            ax.scatter(
                rep_coords[0, ca_indices, 0],
                rep_coords[0, ca_indices, 1],
                rep_coords[0, ca_indices, 2],
                c='blue', label='Original', s=10
            )
            
            # Apply the eigenmode with scaling
            mode_effect = np.zeros_like(rep_coords[0])
            mode_effect.flat[:] = centroid * 0.5  # Scale factor
            
            # Plot structure with eigenmode applied
            modified_coords = rep_coords[0] + mode_effect
            ax.scatter(
                modified_coords[ca_indices, 0],
                modified_coords[ca_indices, 1],
                modified_coords[ca_indices, 2],
                c='red', label='Eigenmode Applied', s=10
            )
            
            ax.set_title(f"Cluster {i+1}: {cluster_id} (n={cluster_data['sample_count']})")
            ax.legend()
            
            plt.tight_layout()
            
            # Store visualization
            visualizations[cluster_id] = {
                'figure': fig,
                'domain': rep_domain,
                'temperature': rep_temp,
                'centroid': centroid
            }
        
        return visualizations