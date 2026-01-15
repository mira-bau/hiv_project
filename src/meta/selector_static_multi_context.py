"""
Static Multi-Context Meta-Selector for Policy Selection

This module implements a domain-agnostic meta-selector that processes per-timestep features
(state, policy signals, CF signals) as tokens in a Transformer architecture, without using
temporal sequences.

The architecture uses a token-based Transformer:
- Tokenization: CLS + state + CF + 5 policy tokens = 7 tokens
- Transformer encoder processes the token sequence
- Output head produces logits over policies
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. StaticMultiContextMetaSelector will not work.")


class StaticMultiContextMetaSelector:
    """
    Static multi-context meta-selector using token-based Transformer architecture.
    
    Processes three input groups:
    - State features: (B, state_dim) - per-timestep state vector
    - Per-policy features: (B, num_policies, policy_feat_dim) - policy signals
    - CF neighbor features: (B, neighbor_dim) - CF signals
    
    Architecture:
    1. Tokenization: CLS + state + CF + 5 policy tokens = 7 tokens
    2. Positional encoding: learned positional embeddings
    3. Transformer encoder: processes token sequence
    4. Output head: classification to num_policies
    """
    
    def __init__(
        self,
        policy_names: List[str],
        state_dim: int,
        policy_feat_dim: int,
        neighbor_dim: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        num_epochs: int = 25,
        patience: int = 5,
        min_delta: float = 1e-4,
        use_class_weights: bool = True,
        use_cf_token: bool = True,
        device: str = "cpu",
        verbose: bool = True
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        self.policy_names = policy_names
        self.state_dim = state_dim
        self.policy_feat_dim = policy_feat_dim
        self.neighbor_dim = neighbor_dim
        self.use_cf_token = use_cf_token
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.use_class_weights = use_class_weights
        self.device = device
        self.verbose = verbose
        
        self.policy_to_idx = {name: idx for idx, name in enumerate(policy_names)}
        self.n_policies = len(policy_names)
        
        self.model = None
        self.best_model_state = None
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': [], 'epoch': []
        }
    
    def _create_model(self):
        """Create StaticMultiContextMetaSelector model architecture."""
        return StaticMultiContextModel(
            state_dim=self.state_dim,
            policy_feat_dim=self.policy_feat_dim,
            neighbor_dim=self.neighbor_dim,
            num_policies=self.n_policies,
            use_cf_token=self.use_cf_token,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
    
    def fit(
        self,
        state_x: np.ndarray,
        policy_feats: np.ndarray,
        neighbor_feats: np.ndarray,
        best_policy: np.ndarray,
        val_split: float = 0.1
    ):
        """
        Train StaticMultiContextMetaSelector.
        
        Args:
            state_x: State features (n_samples, state_dim)
            policy_feats: Per-policy features (n_samples, num_policies, policy_feat_dim)
            neighbor_feats: CF neighbor features (n_samples, neighbor_dim)
            best_policy: Best policy indices (n_samples,)
            val_split: Fraction of data for validation
        """
        start_time = time.time()
        
        # Create model
        if self.model is None:
            self.model = self._create_model()
        
        # Compute class weights if enabled
        class_weights = None
        if self.use_class_weights:
            class_counts = np.bincount(best_policy, minlength=self.n_policies)
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.mean()  # Normalize by mean
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            
            if self.verbose:
                print(f"[STATIC META] Train label distribution (indices): {class_counts.tolist()}")
                label_dist_names = {self.policy_names[i] if i < len(self.policy_names) else f"unknown_{i}": int(count) 
                                   for i, count in enumerate(class_counts)}
                print(f"[STATIC META] Train label distribution (names): {label_dist_names}")
                print(f"[STATIC META] Class weights: {class_weights.cpu().numpy().tolist()}")
        
        # Split data
        n_samples = len(state_x)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        
        state_x_train = state_x[indices[n_val:]]
        policy_feats_train = policy_feats[indices[n_val:]]
        neighbor_feats_train = neighbor_feats[indices[n_val:]]
        best_policy_train = best_policy[indices[n_val:]]
        
        state_x_val = state_x[indices[:n_val]]
        policy_feats_val = policy_feats[indices[:n_val]]
        neighbor_feats_val = neighbor_feats[indices[:n_val]]
        best_policy_val = best_policy[indices[:n_val]]
        
        # Create datasets
        train_dataset = StaticMultiContextDataset(
            state_x_train, policy_feats_train, neighbor_feats_train, best_policy_train
        )
        val_dataset = StaticMultiContextDataset(
            state_x_val, policy_feats_val, neighbor_feats_val, best_policy_val
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        
        # Early stopping setup
        best_val_loss = np.inf
        patience_counter = 0
        best_epoch = 0
        
        if self.verbose:
            print(f"[STATIC META] Training on {len(state_x_train)} samples, validating on {len(state_x_val)} samples")
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_state, batch_policy, batch_neighbor, batch_y in train_loader:
                batch_state = batch_state.to(self.device)
                batch_policy = batch_policy.to(self.device)
                batch_neighbor = batch_neighbor.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_state, batch_policy, batch_neighbor)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_state, batch_policy, batch_neighbor, batch_y in val_loader:
                    batch_state = batch_state.to(self.device)
                    batch_policy = batch_policy.to(self.device)
                    batch_neighbor = batch_neighbor.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_state, batch_policy, batch_neighbor)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # LR Scheduler step
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['lr'].append(current_lr)
            self.training_history['epoch'].append(epoch + 1)
            
            # Logging
            if self.verbose and (epoch % 5 == 0 or epoch == self.num_epochs - 1 or epoch < 3):
                print(f"[STATIC META] Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                      f"LR: {current_lr:.6f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss - self.min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                if self.verbose and epoch > 0:
                    print(f"[STATIC META] ✓ New best Val Loss: {avg_val_loss:.4f} at epoch {best_epoch}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"[STATIC META] Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"[STATIC META] Loaded best model from epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
        
        training_time = time.time() - start_time
        if self.verbose:
            print(f"[STATIC META] Training complete in {training_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")
        
        return self
    
    def select(
        self,
        state_x: np.ndarray,
        policy_feats: np.ndarray,
        neighbor_feats: np.ndarray
    ) -> str:
        """
        Select best policy for given inputs.
        
        Args:
            state_x: State features (1, state_dim) or (state_dim,)
            policy_feats: Per-policy features (1, num_policies, policy_feat_dim) or (num_policies, policy_feat_dim)
            neighbor_feats: CF neighbor features (1, neighbor_dim) or (neighbor_dim,)
        
        Returns:
            Name of selected policy
        """
        # Ensure batch dimension
        if state_x.ndim == 1:
            state_x = state_x.reshape(1, -1)
        if policy_feats.ndim == 2:
            policy_feats = policy_feats.reshape(1, *policy_feats.shape)
        if neighbor_feats.ndim == 1:
            neighbor_feats = neighbor_feats.reshape(1, -1)
        
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_x).to(self.device)
            policy_tensor = torch.FloatTensor(policy_feats).to(self.device)
            neighbor_tensor = torch.FloatTensor(neighbor_feats).to(self.device)
            
            outputs = self.model(state_tensor, policy_tensor, neighbor_tensor)
            _, predicted = torch.max(outputs, 1)
            idx = int(predicted[0].cpu().numpy())
        
        return self.policy_names[idx]
    
    def get_policy_probabilities(
        self,
        state_x: np.ndarray,
        policy_feats: np.ndarray,
        neighbor_feats: np.ndarray
    ) -> Dict[str, float]:
        """Get probability distribution over policies."""
        # Ensure batch dimension
        if state_x.ndim == 1:
            state_x = state_x.reshape(1, -1)
        if policy_feats.ndim == 2:
            policy_feats = policy_feats.reshape(1, *policy_feats.shape)
        if neighbor_feats.ndim == 1:
            neighbor_feats = neighbor_feats.reshape(1, -1)
        
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_x).to(self.device)
            policy_tensor = torch.FloatTensor(policy_feats).to(self.device)
            neighbor_tensor = torch.FloatTensor(neighbor_feats).to(self.device)
            
            outputs = self.model(state_tensor, policy_tensor, neighbor_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        return {self.policy_names[i]: float(probs[i]) for i in range(len(self.policy_names))}


class StaticMultiContextModel(nn.Module):
    """
    Static multi-context Transformer-based policy selector.
    
    Architecture:
    1. Tokenization: CLS + state + ([CF] if use_cf_token) + num_policies (policy tokens)
    2. Positional encoding: learned positional embeddings
    3. Transformer encoder: processes token sequence
    4. Output head: classification to num_policies
    """
    
    def __init__(
        self,
        state_dim: int,
        policy_feat_dim: int,
        neighbor_dim: int,
        num_policies: int,
        use_cf_token: bool = True,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.15
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.policy_feat_dim = policy_feat_dim
        self.neighbor_dim = neighbor_dim
        self.num_policies = num_policies
        self.use_cf_token = use_cf_token
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
        # Total tokens: 1 (CLS) + 1 (state) + (1 if use_cf_token else 0) (CF) + num_policies (policies)
        self.num_tokens = num_policies + 2 + (1 if use_cf_token else 0)
        
        # Token embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # CLS token
        self.state_embed = nn.Linear(state_dim, d_model)  # State token
        if use_cf_token:
            self.neighbor_embed = nn.Linear(neighbor_dim, d_model)  # CF token
        self.policy_embed = nn.Linear(policy_feat_dim, d_model)  # Policy tokens (shared)
        
        # Positional encoding (learned)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        
        # Output head
        self.out_head = nn.Linear(d_model, num_policies)
        
        # Track if we've warned about missing attributes (to avoid spam)
        self._warned_missing_use_cf_token = False
        self._warned_missing_neighbor_embed = False
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Initialize CLS token and positional embeddings
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(
        self,
        state_x: torch.Tensor,
        policy_feats: torch.Tensor,
        neighbor_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state_x: State features (B, state_dim)
            policy_feats: Per-policy features (B, num_policies, policy_feat_dim)
            neighbor_feats: CF neighbor features (B, neighbor_dim) - ignored if use_cf_token=False
        
        Returns:
            logits: Policy logits (B, num_policies)
        """
        B = state_x.size(0)
        
        # 1. Tokenization
        # CLS token
        cls = self.cls_token.expand(B, 1, self.d_model)  # (B, 1, d_model)
        
        # State token
        state_tok = self.state_embed(state_x).unsqueeze(1)  # (B, 1, d_model)
        
        # Policy tokens
        policy_tokens = self.policy_embed(policy_feats)  # (B, num_policies, d_model)
        
        # CF token (conditional)
        # Use getattr to handle models saved before use_cf_token was added
        use_cf_token = getattr(self, 'use_cf_token', True)
        if not hasattr(self, 'use_cf_token') and not getattr(self, '_warned_missing_use_cf_token', False):
            print("[STATIC META] Warning: Model missing 'use_cf_token' attribute. Using default: True")
            self._warned_missing_use_cf_token = True
        
        # Also check if neighbor_embed exists (in case model was saved with use_cf_token=False)
        has_neighbor_embed = hasattr(self, 'neighbor_embed')
        if use_cf_token and not has_neighbor_embed and not getattr(self, '_warned_missing_neighbor_embed', False):
            print("[STATIC META] Warning: Model has use_cf_token=True but missing 'neighbor_embed' layer. "
                  "Falling back to mode without CF token.")
            self._warned_missing_neighbor_embed = True
        
        if use_cf_token and has_neighbor_embed:
            neighbor_tok = self.neighbor_embed(neighbor_feats).unsqueeze(1)  # (B, 1, d_model)
            # Concatenate: [CLS, state, CF, policy_0, policy_1, ..., policy_{num_policies-1}]
            tokens = torch.cat([cls, state_tok, neighbor_tok, policy_tokens], dim=1)
            # Shape: (B, 1 + 1 + 1 + num_policies, d_model) = (B, num_policies + 3, d_model)
        else:
            # Concatenate: [CLS, state, policy_0, policy_1, ..., policy_{num_policies-1}]
            tokens = torch.cat([cls, state_tok, policy_tokens], dim=1)
            # Shape: (B, 1 + 1 + num_policies, d_model) = (B, num_policies + 2, d_model)
        
        # 2. Add positional encoding
        tokens = tokens + self.pos_embed  # (B, num_tokens, d_model)
        
        # 3. Transformer encoder
        encoded = self.transformer(tokens)  # (B, num_tokens, d_model)
        
        # 4. Take CLS token output (first token)
        cls_output = encoded[:, 0, :]  # (B, d_model)
        
        # 5. Output head
        logits = self.out_head(cls_output)  # (B, num_policies)
        
        return logits


class StaticMultiContextDataset(Dataset):
    """PyTorch dataset for static multi-context inputs."""
    
    def __init__(
        self,
        state_x: np.ndarray,
        policy_feats: np.ndarray,
        neighbor_feats: np.ndarray,
        labels: np.ndarray
    ):
        self.state_x = torch.FloatTensor(state_x)
        self.policy_feats = torch.FloatTensor(policy_feats)
        self.neighbor_feats = torch.FloatTensor(neighbor_feats)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.state_x)
    
    def __getitem__(self, idx):
        return (
            self.state_x[idx],
            self.policy_feats[idx],
            self.neighbor_feats[idx],
            self.labels[idx]
        )


if __name__ == "__main__":
    """Runtime check for both use_cf_token modes."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping runtime check")
    else:
        print("Running runtime check for StaticMultiContextModel...")
        
        # Test parameters
        batch_size = 2
        state_dim = 4
        policy_feat_dim = 2
        neighbor_dim = 3
        num_policies = 5
        d_model = 64
        
        # Test with CF token (default)
        print("\n1. Testing with use_cf_token=True...")
        model_with_cf = StaticMultiContextModel(
            state_dim=state_dim,
            policy_feat_dim=policy_feat_dim,
            neighbor_dim=neighbor_dim,
            num_policies=num_policies,
            use_cf_token=True,
            d_model=d_model
        )
        
        state_x = torch.randn(batch_size, state_dim)
        policy_feats = torch.randn(batch_size, num_policies, policy_feat_dim)
        neighbor_feats = torch.randn(batch_size, neighbor_dim)
        
        with torch.no_grad():
            logits = model_with_cf(state_x, policy_feats, neighbor_feats)
        print(f"   Input shapes: state={state_x.shape}, policy={policy_feats.shape}, neighbor={neighbor_feats.shape}")
        print(f"   Output shape: {logits.shape}")
        print(f"   Expected tokens: {num_policies + 3} (CLS + state + CF + {num_policies} policies)")
        print(f"   Actual num_tokens: {model_with_cf.num_tokens}")
        assert logits.shape == (batch_size, num_policies), f"Expected ({batch_size}, {num_policies}), got {logits.shape}"
        assert model_with_cf.num_tokens == num_policies + 3, f"Expected {num_policies + 3}, got {model_with_cf.num_tokens}"
        print("   ✓ use_cf_token=True test passed")
        
        # Test without CF token
        print("\n2. Testing with use_cf_token=False...")
        model_no_cf = StaticMultiContextModel(
            state_dim=state_dim,
            policy_feat_dim=policy_feat_dim,
            neighbor_dim=neighbor_dim,  # Still required but ignored
            num_policies=num_policies,
            use_cf_token=False,
            d_model=d_model
        )
        
        with torch.no_grad():
            logits = model_no_cf(state_x, policy_feats, neighbor_feats)
        print(f"   Input shapes: state={state_x.shape}, policy={policy_feats.shape}, neighbor={neighbor_feats.shape}")
        print(f"   Output shape: {logits.shape}")
        print(f"   Expected tokens: {num_policies + 2} (CLS + state + {num_policies} policies)")
        print(f"   Actual num_tokens: {model_no_cf.num_tokens}")
        assert logits.shape == (batch_size, num_policies), f"Expected ({batch_size}, {num_policies}), got {logits.shape}"
        assert model_no_cf.num_tokens == num_policies + 2, f"Expected {num_policies + 2}, got {model_no_cf.num_tokens}"
        print("   ✓ use_cf_token=False test passed")
        
        print("\n✓ All runtime checks passed!")


