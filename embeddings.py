"""
Extract embeddings and contrastive embeddings from trained AbAgCDM model.
Saves embeddings with ab_id and ag_id for later analysis.
""" 
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Literal, Optional
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore') 

from AbAgCDM import get_AbAgCDM, create_single_dataloader  


# ===== EXTRACT EMBEDDINGS ===== 
def extract_all_embeddings(
    model,
    dataloader,
    device: torch.device,
    save_dir: str = "./embeddings",
) -> Dict[str, np.ndarray]:
    """
    Extract regular and contrastive embeddings from the model.
    
    Args:
        model: Trained AbAgCDM model
        dataloader: DataLoader with encoded sequences (batch contains ab_id, ag_id)
        device: Device to run inference on
        save_dir: Directory to save embeddings
    
    Returns:
        Dictionary with all embeddings and metadata 
    """
    model.eval()
    model.to(device)
    
    # Storage for embeddings
    all_embeddings = {
        'ab_id': [],
        'ag_id': [],
        'binding_label': [],
        # Regular embeddings
        'emb_cls': [],
        'emb_vhh': [],
        'emb_il6': [],
        # Contrastive embeddings
        'z_vhh': [],
        'z_il6': [],
    } 
    
    print(f"Extracting embeddings from {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            il6_start_idx = batch['il6_start_idx'].to(device)
            il6_end_idx = batch['il6_end_idx'].to(device)
            
            batch_size = input_ids.shape[0]
            
            # Get regular embeddings
            embeddings = model.get_embeddings(
                input_ids=input_ids,
                il6_start_idx=il6_start_idx,
                il6_end_idx=il6_end_idx,
                attention_mask=attention_mask,
                return_token_embeddings=False
            )
            
            # Get contrastive embeddings
            contrastive_embeddings = model.get_contrastive_embeddings(
                input_ids=input_ids,
                il6_start_idx=il6_start_idx,
                il6_end_idx=il6_end_idx,
                attention_mask=attention_mask
            )
            
            # Store metadata from batch
            all_embeddings['ab_id'].extend(batch['ab_id'])
            all_embeddings['ag_id'].extend(batch['ag_id'])
            all_embeddings['binding_label'].extend(
                batch['binding_label'].cpu().numpy().tolist()
            )
            
            # Store regular embeddings (move to CPU and convert to numpy)
            all_embeddings['emb_cls'].append(embeddings['cls'].cpu().numpy())
            all_embeddings['emb_vhh'].append(embeddings['vhh'].cpu().numpy())
            all_embeddings['emb_il6'].append(embeddings['il6'].cpu().numpy())
            
            # Store contrastive embeddings
            all_embeddings['z_vhh'].append(contrastive_embeddings['vhh'].cpu().numpy())
            all_embeddings['z_il6'].append(contrastive_embeddings['il6'].cpu().numpy())
    
    
    # Concatenate all embeddings
    print("Concatenating embeddings...")
    all_embeddings['emb_cls'] = np.vstack(all_embeddings['emb_cls'])
    all_embeddings['emb_vhh'] = np.vstack(all_embeddings['emb_vhh'])
    all_embeddings['emb_il6'] = np.vstack(all_embeddings['emb_il6'])
    all_embeddings['z_vhh'] = np.vstack(all_embeddings['z_vhh'])
    all_embeddings['z_il6'] = np.vstack(all_embeddings['z_il6'])
    
    # Convert lists to arrays
    all_embeddings['ab_id'] = np.array(all_embeddings['ab_id'])
    all_embeddings['ag_id'] = np.array(all_embeddings['ag_id'])
    all_embeddings['binding_label'] = np.array(all_embeddings['binding_label'])
    
    # Print summary
    print(f"\nExtracted embeddings for {len(all_embeddings['ab_id'])} samples")
    print(f"  CLS embeddings: {all_embeddings['emb_cls'].shape}")
    print(f"  VHH embeddings: {all_embeddings['emb_vhh'].shape}")
    print(f"  IL6 embeddings: {all_embeddings['emb_il6'].shape}")
    print(f"  VHH contrastive: {all_embeddings['z_vhh'].shape}")
    print(f"  IL6 contrastive: {all_embeddings['z_il6'].shape}")
    
    # Save embeddings
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle (preserves structure)
    pickle_path = save_path / "embeddings.pkl"
    print(f"\nSaving to {pickle_path}...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_embeddings, f)
    
    # Also save as npz (more portable)
    npz_path = save_path / "embeddings.npz"
    print(f"Saving to {npz_path}...")
    np.savez_compressed(
        npz_path,
        ab_id=all_embeddings['ab_id'],
        ag_id=all_embeddings['ag_id'],
        binding_label=all_embeddings['binding_label'],
        emb_cls=all_embeddings['emb_cls'],
        emb_vhh=all_embeddings['emb_vhh'],
        emb_il6=all_embeddings['emb_il6'],
        z_vhh=all_embeddings['z_vhh'],
        z_il6=all_embeddings['z_il6']
    )
    
    # Save metadata as CSV for easy inspection
    csv_path = save_path / "embeddings_metadata.csv"
    print(f"Saving metadata to {csv_path}...")
    metadata_df = pd.DataFrame({
        'ab_id': all_embeddings['ab_id'],
        'ag_id': all_embeddings['ag_id'],
        'binding_label': all_embeddings['binding_label']
    })
    metadata_df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Embeddings saved successfully!")
    print(f"  - Full data: {pickle_path}")
    print(f"  - Compressed: {npz_path}")
    print(f"  - Metadata: {csv_path}")
    
    return all_embeddings


# ===== LOAD SAVED EMBEDDINGS ===== 
def load_embeddings(save_dir: str = "./embeddings") -> Dict[str, np.ndarray]:
    """
    Load previously saved embeddings.
    
    Args:
        save_dir: Directory where embeddings were saved
    
    Returns:
        Dictionary with all embeddings and metadata
    """
    save_path = Path(save_dir)
    
    # Try pickle first (preserves exact structure)
    pickle_path = save_path / "embeddings.pkl"
    if pickle_path.exists():
        print(f"Loading embeddings from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"✓ Loaded {len(embeddings['ab_id'])} samples")
        return embeddings
    
    # Fall back to npz
    npz_path = save_path / "embeddings.npz"
    if npz_path.exists():
        print(f"Loading embeddings from {npz_path}...")
        data = np.load(npz_path)
        embeddings = {key: data[key] for key in data.files}
        print(f"✓ Loaded {len(embeddings['ab_id'])} samples")
        return embeddings
    
    raise FileNotFoundError(f"No embeddings found in {save_dir}") 


def plot_embeddings(
    embeddings: Dict[str, np.ndarray],
    embedding_type: Literal['emb_cls', 'emb_vhh', 'emb_il6', 'z_vhh', 'z_il6'] = 'emb_cls',
    label_by: Literal['binding_label', 'ab_id', 'ag_id'] = 'binding_label',
    method: Literal['tsne', 'umap'] = 'tsne',
    save_path: Optional[str] = None,
    n_samples: Optional[int] = None,
    use_pca: Optional[bool] = None,
    pca_components: Optional[int] = None,
    perplexity: int = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    figsize: tuple = (12, 8),
    alpha: float = 0.6,
    s: int = 20,
    show_legend: bool = True,
    title: Optional[str] = None,
    show_pca_variance: bool = True 
) -> tuple:
    """
    Visualize embeddings using t-SNE or UMAP.
    
    Args:
        embeddings: Dictionary from load_embeddings()
        embedding_type: Which embedding to plot ('emb_cls', 'emb_vhh', 'emb_il6', 'z_vhh', 'z_il6')
        label_by: How to color points ('binding_label', 'ab_id', 'ag_id')
        method: Dimensionality reduction method ('tsne' or 'umap')
        save_path: Path to save figure (if None, displays plot)
        n_samples: Number of samples to plot (if None, uses all)
        use_pca: Whether to use PCA preprocessing (if None, auto: True for dim>100)
        pca_components: Number of PCA components (if None, auto: 50 for ESM-2, 30 for contrastive)
        perplexity: t-SNE perplexity parameter
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed for reproducibility
        figsize: Figure size (width, height)
        alpha: Point transparency
        s: Point size
        show_legend: Whether to show legend
        title: Custom title (if None, auto-generates)
        show_pca_variance: Whether to print PCA variance explained 
    
    Returns:
        (reduced_embeddings, labels, fig, ax)
    """
    # Validate inputs
    if embedding_type not in embeddings:
        raise ValueError(f"embedding_type '{embedding_type}' not found. Available: {list(embeddings.keys())}")
    if label_by not in embeddings:
        raise ValueError(f"label_by '{label_by}' not found. Available: {list(embeddings.keys())}")
    if method == 'umap' and not UMAP_AVAILABLE:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    
    # Get data
    X = embeddings[embedding_type]
    labels = embeddings[label_by]
    
    # Sample if requested
    if n_samples is not None and n_samples < len(X):
        print(f"Sampling {n_samples} from {len(X)} total samples...")
        idx = np.random.RandomState(random_state).choice(len(X), n_samples, replace=False)
        X = X[idx]
        labels = labels[idx]
    original_dim = X.shape[1] 
    print(f"Reducing {X.shape} embeddings using {method.upper()}...")
    
    # Determine PCA settings
    if use_pca is None:
        # Auto-enable PCA for high dimensions
        use_pca = original_dim > 100
    
    if use_pca:
        if pca_components is None:
            # Auto-determine number of components
            if original_dim > 500:  # ESM-2 embeddings (~1280D)
                pca_components = 50
            elif original_dim > 200:  # Contrastive embeddings (~256D)
                pca_components = 30
            else:
                pca_components = min(50, original_dim // 2)
        
        # Ensure valid number of components
        pca_components = min(pca_components, min(X.shape[0], X.shape[1]) - 1)
        
        print(f"Applying PCA: {original_dim}D → {pca_components}D...")
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_pca = pca.fit_transform(X)
        
        variance_explained = pca.explained_variance_ratio_.sum() * 100
        if show_pca_variance:
            print(f"  ✓ PCA complete: {variance_explained:.2f}% variance explained")
            # Show cumulative variance for first few components
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            for i in [5, 10, 20, 30, 40, 50]:
                if i <= pca_components:
                    print(f"    First {i} components: {cumsum[i-1]*100:.2f}% variance")
        
        X_input = X_pca
        pca_model = pca
    else:
        print("Skipping PCA preprocessing")
        X_input = X
        pca_model = None
    
    print(f"Reducing {X_input.shape} embeddings using {method.upper()}...")
    
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(X_input) - 1),
            random_state=random_state,
            n_jobs=-1
        )
    else:  # umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(X_input) - 1),
            min_dist=min_dist,
            random_state=random_state
        )
    
    X_reduced = reducer.fit_transform(X_input)
    print(f"✓ Reduction complete: {X_reduced.shape}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine coloring scheme
    unique_labels = np.unique(labels)
    n_unique = len(unique_labels)
    
    if label_by == 'binding_label':
        # Binary: use red/blue
        colors = ['#e74c3c', '#3498db']  # red for 0, blue for 1 
        cmap = ListedColormap(colors)
        scatter = ax.scatter(
            X_reduced[:, 0], X_reduced[:, 1],
            c=labels, cmap=cmap,
            alpha=alpha, s=s, edgecolors='none'
        )
        legend_labels = ['Non-binding (0)', 'Binding (1)']
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=colors[i], markersize=10, label=legend_labels[i])
                  for i in range(2)]
        
    elif n_unique <= 20:
        # Categorical with <= 20 classes: use distinct colors
        palette = sns.color_palette('tab20', n_unique) if n_unique > 10 else sns.color_palette('tab10', n_unique)
        label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}
        colors_array = np.array([label_to_color[label] for label in labels])
        
        scatter = ax.scatter(
            X_reduced[:, 0], X_reduced[:, 1],
            c=colors_array,
            alpha=alpha, s=s, edgecolors='none'
        )
        
        if show_legend:
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=label_to_color[label], markersize=10, label=str(label))
                      for label in unique_labels]
        
    else:
        # Too many categories: use continuous colormap
        print(f"Warning: {n_unique} unique labels is too many for discrete colors. Using continuous colormap.")
        # Map labels to numeric values
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_to_idx[label] for label in labels])
        
        scatter = ax.scatter(
            X_reduced[:, 0], X_reduced[:, 1],
            c=numeric_labels, cmap='viridis',
            alpha=alpha, s=s, edgecolors='none'
        )
        if show_legend:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(label_by, rotation=270, labelpad=20)
        handles = None
    
    # Add legend
    if show_legend and handles is not None and n_unique <= 20:
        legend = ax.legend(
            handles=handles,
            title=label_by,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=True,
            fontsize=9
        )
        legend.get_title().set_fontsize(10)
    
    # Labels and title
    ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
    
    if title is None:
        pca_note = f" (PCA→{pca_components}D)" if use_pca else ""
        title = f'{embedding_type} embeddings{pca_note} ({method.upper()})\nColored by {label_by}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add PCA variance note if used
    if use_pca and show_pca_variance:
        variance_text = f'PCA: {variance_explained:.1f}% variance'
        ax.text(0.02, 0.98, variance_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Style
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    fig.tight_layout()
    
    # Save or show
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    else:
        plt.show()
    
    return X_reduced, labels, fig, ax, pca_model 


def plot_all_embeddings(
    embeddings: Dict[str, np.ndarray],
    label_by: Literal['binding_label', 'ab_id', 'ag_id'] = 'binding_label',
    method: Literal['tsne', 'umap'] = 'tsne',
    save_dir: Optional[str] = None,
    n_samples: Optional[int] = None,
    **kwargs
) -> Dict[str, tuple]:
    """
    Plot all embedding types in one go.
    
    Args:
        embeddings: Dictionary from load_embeddings()
        label_by: How to color points
        method: Dimensionality reduction method
        save_dir: Directory to save all plots
        n_samples: Number of samples to plot per embedding type
        **kwargs: Additional arguments for plot_embeddings()
    
    Returns:
        Dictionary mapping embedding_type to (reduced_embeddings, labels, fig, ax)
    """
    embedding_types = ['emb_cls', 'emb_vhh', 'emb_il6', 'z_vhh', 'z_il6']
    results = {}
    
    print(f"\nPlotting all {len(embedding_types)} embedding types...")
    print("="*80)
    
    for emb_type in embedding_types:
        if emb_type not in embeddings:
            print(f"⚠ Skipping {emb_type} (not found in embeddings)")
            continue
        
        print(f"\n[{emb_type}]")
        
        # Determine save path
        if save_dir:
            save_path = Path(save_dir) / f"{emb_type}_{method}_{label_by}.png"
        else:
            save_path = None
        
        # Plot
        result = plot_embeddings(
            embeddings=embeddings,
            embedding_type=emb_type,
            label_by=label_by,
            method=method,
            save_path=save_path,
            n_samples=n_samples,
            **kwargs
        )
        results[emb_type] = result
        
        # Close figure to save memory
        if save_dir:
            plt.close(result[2])
    
    print("\n" + "="*80)
    print(f"✓ Completed plotting {len(results)} embedding types")
    
    return results


def compare_embedding_methods(
    embeddings: Dict[str, np.ndarray],
    embedding_type: Literal['emb_cls', 'emb_vhh', 'emb_il6', 'z_vhh', 'z_il6'] = 'emb_cls',
    label_by: Literal['binding_label', 'ab_id', 'ag_id'] = 'binding_label',
    save_path: Optional[str] = None,
    n_samples: Optional[int] = None,
    use_pca: Optional[bool] = None,
    pca_components: Optional[int] = None,
    show_pca_variance: bool = True,
    **kwargs
) -> tuple:
    """
    Compare t-SNE and UMAP side-by-side for one embedding type with optional PCA.
    
    Args:
        embeddings: Dictionary from load_embeddings()
        embedding_type: Which embedding to plot
        label_by: How to color points
        save_path: Path to save comparison figure
        n_samples: Number of samples to plot
        use_pca: Whether to use PCA preprocessing (if None, auto: True for dim>100)
        pca_components: Number of PCA components (if None, auto-determined)
        show_pca_variance: Whether to print PCA variance explained
        **kwargs: Additional arguments (perplexity, n_neighbors, alpha, s, random_state, etc.)
    
    Returns:
        (fig, axes, tsne_result, umap_result, pca_model)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    
    print(f"\nComparing t-SNE vs UMAP for {embedding_type}...")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Get data
    X = embeddings[embedding_type]
    labels = embeddings[label_by]
    
    random_state = kwargs.get('random_state', 42)
    
    # Sample if requested
    if n_samples is not None and n_samples < len(X):
        print(f"Sampling {n_samples} from {len(X)} total samples...")
        idx = np.random.RandomState(random_state).choice(len(X), n_samples, replace=False)
        X = X[idx]
        labels = labels[idx]
    
    original_dim = X.shape[1]
    print(f"Starting with {X.shape} embeddings (dim={original_dim})")
    
    # Determine PCA settings
    if use_pca is None:
        use_pca = original_dim > 100
    
    pca_model = None
    if use_pca:
        if pca_components is None:
            if original_dim > 500:
                pca_components = 50
            elif original_dim > 200:
                pca_components = 30
            else:
                pca_components = min(50, original_dim // 2)
        
        pca_components = min(pca_components, min(X.shape[0], X.shape[1]) - 1)
        
        print(f"Applying PCA: {original_dim}D → {pca_components}D...")
        pca_model = PCA(n_components=pca_components, random_state=random_state)
        X = pca_model.fit_transform(X)
        
        variance_explained = pca_model.explained_variance_ratio_.sum() * 100
        if show_pca_variance:
            print(f"  ✓ PCA complete: {variance_explained:.2f}% variance explained")
    else:
        print("Skipping PCA preprocessing")
        variance_explained = None
    
    methods = ['tsne', 'umap']
    results = {}
    
    for i, method in enumerate(methods):
        print(f"\nReducing with {method.upper()}...")
        
        # Reduce dimensions
        if method == 'tsne':
            reducer = TSNE(
                n_components=2,
                perplexity=min(kwargs.get('perplexity', 30), len(X) - 1),
                random_state=random_state,
                n_jobs=-1
            )
        else:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(kwargs.get('n_neighbors', 15), len(X) - 1),
                min_dist=kwargs.get('min_dist', 0.1),
                random_state=random_state
            )
        
        X_reduced = reducer.fit_transform(X)
        print(f"  ✓ Complete: {X_reduced.shape}")
        
        # Plot on subplot
        ax = axes[i]
        
        unique_labels = np.unique(labels)
        n_unique = len(unique_labels)
        
        if label_by == 'binding_label':
            colors = ['#3498db', '#e74c3c']
            cmap = ListedColormap(colors)
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap=cmap,
                               alpha=kwargs.get('alpha', 0.6), s=kwargs.get('s', 20), edgecolors='none')
            if i == 1:  # Only show legend once
                legend_labels = ['Non-binding (0)', 'Binding (1)']
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=colors[j], markersize=10, label=legend_labels[j])
                          for j in range(2)]
                ax.legend(handles=handles, title=label_by, loc='best')
        
        elif n_unique <= 20:
            palette = sns.color_palette('tab20', n_unique) if n_unique > 10 else sns.color_palette('tab10', n_unique)
            label_to_color = {label: palette[j] for j, label in enumerate(unique_labels)}
            colors_array = np.array([label_to_color[label] for label in labels])
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=colors_array,
                               alpha=kwargs.get('alpha', 0.6), s=kwargs.get('s', 20), edgecolors='none')
            if i == 1:
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=label_to_color[label], markersize=10, label=str(label))
                          for label in unique_labels]
                ax.legend(handles=handles, title=label_by, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            label_to_idx = {label: j for j, label in enumerate(unique_labels)}
            numeric_labels = np.array([label_to_idx[label] for label in labels])
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=numeric_labels, cmap='viridis',
                               alpha=kwargs.get('alpha', 0.6), s=kwargs.get('s', 20), edgecolors='none')
        
        ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
        ax.set_title(f'{method.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#f8f9fa')
        
        # Add PCA variance note
        if use_pca and variance_explained and show_pca_variance:
            variance_text = f'PCA: {variance_explained:.1f}% var'
            ax.text(0.02, 0.98, variance_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        results[method] = (X_reduced, labels)
    
    pca_note = f" (PCA→{pca_components}D)" if use_pca else ""
    fig.suptitle(f'{embedding_type} embeddings{pca_note} - Colored by {label_by}', 
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"\n✓ Saved comparison to {save_path}")
    else:
        plt.show()
    
    return fig, axes, results['tsne'], results['umap'], pca_model



def analyze_pca_variance(
    embeddings: Dict[str, np.ndarray],
    embedding_type: Literal['emb_cls', 'emb_vhh', 'emb_il6', 'z_vhh', 'z_il6'] = 'emb_cls',
    max_components: int = 100,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
) -> PCA:
    """
    Analyze PCA variance explained to help choose optimal n_components.
    
    Args:
        embeddings: Dictionary from load_embeddings()
        embedding_type: Which embedding to analyze
        max_components: Maximum number of components to analyze
        save_path: Path to save variance plot
        figsize: Figure size
    
    Returns:
        Fitted PCA model
    """
    X = embeddings[embedding_type]
    
    # Determine number of components
    n_components = min(max_components, min(X.shape) - 1)
    
    print(f"Analyzing PCA variance for {embedding_type}...")
    print(f"  Original dimensions: {X.shape}")
    print(f"  Computing {n_components} components...")
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # Calculate cumulative variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PCA VARIANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"{'Components':<15} {'Variance Explained':<25} {'Cumulative':<15}")
    print(f"{'-'*60}")
    
    milestones = [1, 5, 10, 20, 30, 50, 75, 100]
    for n in milestones:
        if n <= n_components:
            var = pca.explained_variance_ratio_[n-1] * 100
            cum = cumvar[n-1] * 100
            print(f"{n:<15} {var:>8.3f}%{'':<16} {cum:>8.3f}%")
    
    # Find components for different thresholds
    print(f"\n{'='*60}")
    print("COMPONENTS NEEDED FOR VARIANCE THRESHOLDS")
    print(f"{'='*60}")
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
    for threshold in thresholds:
        n_needed = np.argmax(cumvar >= threshold) + 1
        if cumvar[n_needed-1] >= threshold:
            print(f"  {threshold*100:.0f}% variance: {n_needed} components")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Individual variance explained
    components = np.arange(1, n_components + 1)
    ax1.bar(components[:50], pca.explained_variance_ratio_[:50] * 100, 
            color='steelblue', alpha=0.7)
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Variance Explained (%)', fontsize=11)
    ax1.set_title('Individual Component Variance', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(0, min(51, n_components + 1))
    
    # Plot 2: Cumulative variance
    ax2.plot(components, cumvar * 100, linewidth=2.5, color='darkred')
    ax2.axhline(y=90, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='90% threshold')
    ax2.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='95% threshold')
    ax2.fill_between(components, 0, cumvar * 100, alpha=0.2, color='darkred')
    ax2.set_xlabel('Number of Components', fontsize=11)
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=11)
    ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, n_components)
    ax2.set_ylim(0, 105)
    
    fig.suptitle(f'PCA Analysis: {embedding_type} ({X.shape[1]}D)', 
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved PCA analysis to {save_path}")
    else:
        plt.show()
    
    print(f"✓ PCA Analysis complete")
    print(f"{'='*60}\n")
    
    return pca


def compute_silhouette_analysis(
    embeddings: Dict[str, np.ndarray],
    embedding_type: Literal['emb_cls', 'emb_vhh', 'emb_il6', 'z_vhh', 'z_il6'] = 'emb_cls',
    use_pca: bool = True,
    n_pca_components: int = 50,
    save_path: Optional[str] = None,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute silhouette score for binding vs non-binding clusters.
    
    Args:
        embeddings: Dictionary from load_embeddings()
        embedding_type: Which embedding to analyze
        use_pca: Whether to apply PCA preprocessing
        n_pca_components: Number of PCA components
        save_path: Path to save silhouette plot
        random_state: Random seed
    
    Returns:
        Dictionary with silhouette scores
    """
    print(f"\n{'='*80}")
    print(f"SILHOUETTE ANALYSIS: {embedding_type}")
    print(f"{'='*80}")
    
    # Get data
    X = embeddings[embedding_type]
    labels = embeddings['binding_label']
    
    # Check if we have both classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"⚠ Only one class present. Cannot compute silhouette score.")
        return {'overall': None}
    
    print(f"Original dimensions: {X.shape}")
    print(f"Class distribution: {np.bincount(labels.astype(int))}")
    
    # PCA preprocessing
    pca_model = None
    if use_pca and X.shape[1] > 100:
        n_pca_components = min(n_pca_components, X.shape[0] - 1, X.shape[1])
        print(f"Applying PCA: {X.shape[1]}D → {n_pca_components}D...")
        pca_model = PCA(n_components=n_pca_components, random_state=random_state)
        X = pca_model.fit_transform(X)
        variance = pca_model.explained_variance_ratio_.sum() * 100
        print(f"  ✓ Variance explained: {variance:.2f}%")
    
    # Compute overall silhouette score
    print(f"\nComputing silhouette scores...")
    overall_score = silhouette_score(X, labels)
    print(f"  Overall silhouette score: {overall_score:.4f}")
    
    # Compute per-sample silhouette scores
    sample_scores = silhouette_samples(X, labels)
    
    # Per-class scores
    scores_class_0 = sample_scores[labels == 0]
    scores_class_1 = sample_scores[labels == 1]
    
    print(f"  Non-binding (0): {scores_class_0.mean():.4f} ± {scores_class_0.std():.4f}")
    print(f"  Binding (1):     {scores_class_1.mean():.4f} ± {scores_class_1.std():.4f}")
    
    results = {
        'overall': overall_score,
        'non_binding_mean': scores_class_0.mean(),
        'non_binding_std': scores_class_0.std(),
        'binding_mean': scores_class_1.mean(),
        'binding_std': scores_class_1.std(),
    }
    
    # Visualization
    if save_path:
        print(f"\nCreating silhouette plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Silhouette plot
        y_lower = 10
        colors = ['#e74c3c', '#3498db']
        
        for i, label in enumerate([0, 1]):
            # Get silhouette scores for this class
            class_scores = sample_scores[labels == label]
            class_scores.sort()
            
            size_cluster = class_scores.shape[0]
            y_upper = y_lower + size_cluster
            
            color = colors[i]
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                class_scores,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster, 
                    f"Class {label}\n({size_cluster} samples)",
                    fontsize=10, va='center', ha='right')
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10
        
        ax1.set_xlabel("Silhouette Coefficient", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Sample Index", fontsize=11, fontweight='bold')
        ax1.set_title("Silhouette Plot by Binding Class", fontsize=12, fontweight='bold')
        
        # The vertical line for average silhouette score
        ax1.axvline(x=overall_score, color="red", linestyle="--", linewidth=2,
                   label=f'Overall score: {overall_score:.3f}')
        ax1.set_xlim([-0.2, 1])
        ax1.set_ylim([0, len(labels) + (len(unique_labels) + 1) * 10])
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Distribution comparison
        ax2.hist(scores_class_0, bins=30, alpha=0.6, color='#e74c3c', 
                label=f'Non-binding (mean={scores_class_0.mean():.3f})', density=True)
        ax2.hist(scores_class_1, bins=30, alpha=0.6, color='#3498db',
                label=f'Binding (mean={scores_class_1.mean():.3f})', density=True)
        
        ax2.axvline(x=overall_score, color="red", linestyle="--", linewidth=2,
                   label=f'Overall: {overall_score:.3f}')
        
        ax2.set_xlabel("Silhouette Coefficient", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Density", fontsize=11, fontweight='bold')
        ax2.set_title("Distribution of Silhouette Scores", fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xlim([-0.2, 1])
        
        pca_note = f" (PCA→{n_pca_components}D)" if use_pca and pca_model else ""
        fig.suptitle(f'Silhouette Analysis: {embedding_type}{pca_note}',
                    fontsize=14, fontweight='bold', y=1.00)
        fig.tight_layout()
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved silhouette plot to {save_path}")
    
    print(f"✓ Silhouette analysis complete")
    print(f"{'='*80}\n")
    
    return results


def compute_all_silhouette_scores(
    embeddings: Dict[str, np.ndarray],
    save_dir: str = "./embeddings",
    use_pca: bool = True,
    n_pca_components: int = 50
) -> pd.DataFrame:
    """
    Compute silhouette scores for all embedding types.
    
    Args:
        embeddings: Dictionary from load_embeddings()
        save_dir: Directory to save results
        use_pca: Whether to apply PCA preprocessing
        n_pca_components: Number of PCA components
    
    Returns:
        DataFrame with silhouette scores for all embedding types
    """
    print(f"\n{'='*80}")
    print("COMPUTING SILHOUETTE SCORES FOR ALL EMBEDDINGS")
    print(f"{'='*80}")
    
    embedding_types = ['emb_cls', 'emb_vhh', 'emb_il6', 'z_vhh', 'z_il6']
    all_results = []
    
    for emb_type in embedding_types:
        if emb_type not in embeddings:
            print(f"⚠ Skipping {emb_type} (not found)")
            continue
        
        print(f"\n[{emb_type}]")
        save_path = Path(save_dir) / "figures" / f"silhouette_{emb_type}.png"
        
        results = compute_silhouette_analysis(
            embeddings=embeddings,
            embedding_type=emb_type,
            use_pca=use_pca,
            n_pca_components=n_pca_components,
            save_path=str(save_path)
        )
        
        results['embedding_type'] = emb_type
        all_results.append(results)
    
    # Create summary DataFrame
    df_results = pd.DataFrame(all_results)
    df_results = df_results[['embedding_type', 'overall', 'non_binding_mean', 
                             'non_binding_std', 'binding_mean', 'binding_std']]
    
    # Save summary
    summary_path = Path(save_dir) / "silhouette_scores_summary.csv"
    df_results.to_csv(summary_path, index=False)
    print(f"\n✓ Saved silhouette scores summary to {summary_path}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SILHOUETTE SCORES SUMMARY")
    print(f"{'='*80}")
    print(df_results.to_string(index=False))
    print(f"{'='*80}\n")
    
    return df_results



# ===== main function 
def emb_extraction_visualization(
    emb_save_dir = "./embeddings/il6_Q144A",
    compute_embeddings = True,
    compute_silhouette = True,
    data_file_pq = None
):
    print("="*50)
    print("EMBEDDING EXTRACTION & VISUALIZATION")
    print("="*50) 
    
    if compute_embeddings and data_file_pq is not None:
        # Load trained model
        print("\n🛑  Loading trained model...")
        trained_model = get_AbAgCDM(model_directory="AbAgCDM",
                                    checkpoint_folder='checkpoints') 
        model = trained_model["model"] 
        device = trained_model["device"]
        model_weights_path = trained_model["weight_path"]
        print(f"✓ Model loaded from {model_weights_path}")
        print(f"✓ Using device: {device}")
        
        # Load data
        print("\n🛑  Loading data...")
        dataloader = create_single_dataloader(
            data_filepath_pq=data_file_pq ,
            batch_size= 32
        )
        print(f"✓ Dataloader batch size: {dataloader.batch_size}") 
        
        # Extract embeddings
        print("\n🛑  Extracting embeddings...")
        all_embeddings = extract_all_embeddings(
            model=model,
            dataloader=dataloader,
            device=device,
            save_dir=emb_save_dir
        )
    
    # Load embeddings 
    print("\n🛑  Testing load functionality...")
    loaded_embeddings = load_embeddings(save_dir=emb_save_dir)
    print(f"✓ Successfully loaded {len(loaded_embeddings['ab_id'])} samples") 
    
    # Compute silhouette scores
    if compute_silhouette:
        print("\n🛑  Computing silhouette scores...")
        silhouette_results = compute_all_silhouette_scores(
            embeddings=loaded_embeddings,
            save_dir=emb_save_dir,
            use_pca=True,
            n_pca_components=50
        )
    
    print("\n🛑  Visualization examples:")    
    print(" --> Analyze PCA variance first (recommended)")
    pca = analyze_pca_variance(loaded_embeddings, 
                               embedding_type='emb_cls', 
                               save_path=emb_save_dir+'/figures/pca_analysis.png')

    print(" --> Plot all embedding types")
    plot_all_embeddings(loaded_embeddings, 
                        label_by='binding_label', 
                        method='tsne', 
                        save_dir=emb_save_dir+'/figures') 

    print(" --> Compare t-SNE vs UMAP")
    compare_embedding_methods(loaded_embeddings, 
                              embedding_type='emb_cls', 
                              label_by='binding_label',
                              save_path=emb_save_dir+'/figures/compare_reduction.png') 
    
    print("\n" + "="*50)
    print("COMPLETE!")
    print("="*50)
    
    
# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__": 
    
    emb_extraction_visualization(
        emb_save_dir="./embeddings/il6_Q144A",
        compute_embeddings=True, 
        compute_silhouette=True,
        data_file_pq="./data/il6_Q144A_disjoint.parquet" 
    )
    
    emb_extraction_visualization(
        emb_save_dir="./embeddings/il6_test",
        compute_embeddings=True, 
        compute_silhouette=True,
        data_file_pq="./data/test.parquet" 
    )
    


