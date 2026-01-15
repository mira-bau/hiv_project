"""
Hybrid Transformer + Gradient Boosting Meta-Selector

This combines the best of both worlds:
- Transformer encoder for feature extraction and temporal modeling
- Gradient Boosting (XGBoost/LightGBM) for final classification

Advantages:
- Transformer learns rich representations
- GB handles non-linear decision boundaries efficiently
- Often achieves better performance than either alone
"""

from typing import List, Dict, Optional
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

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    try:
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True
    except ImportError:
        LIGHTGBM_AVAILABLE = False
    XGBOOST_AVAILABLE = False


class TransformerGBMetaSelector:
    """
    Hybrid Transformer + Gradient Boosting meta-selector.
    
    Architecture:
    1. Transformer encoder extracts embeddings from patient features
    2. Gradient Boosting classifier predicts policy from embeddings
    
    This combines deep learning feature extraction with efficient tree-based classification.
    """
    
    def __init__(
        self,
        policy_names: List[str],
        use_patient_features: bool = True,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.25,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        num_epochs: int = 50,
        patience: int = 8,
        min_delta: float = 1e-4,
        grad_clip: float = 1.0,
        gb_backend: str = "xgboost",  # "xgboost" or "lightgbm"
        gb_params: Optional[Dict] = None,
        device: str = "cpu",
        verbose: bool = True
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        if gb_backend == "xgboost" and not XGBOOST_AVAILABLE:
            if LIGHTGBM_AVAILABLE:
                gb_backend = "lightgbm"
                if verbose:
                    print("[Transformer-GB] XGBoost not available, using LightGBM")
            else:
                raise ImportError("Either XGBoost or LightGBM is required. Install with: pip install xgboost")
        
        self.policy_names = policy_names
        self.use_patient_features = use_patient_features
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip = grad_clip
        self.gb_backend = gb_backend
        self.device = device
        self.verbose = verbose
        
        self.policy_to_idx = {name: idx for idx, name in enumerate(policy_names)}
        self.n_policies = len(policy_names)
        
        # Default GB parameters
        if gb_params is None:
            if gb_backend == "xgboost":
                gb_params = {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'objective': 'multi:softmax',
                    'num_class': self.n_policies,
                    'eval_metric': 'mlogloss',
                    'use_label_encoder': False
                }
            else:  # lightgbm
                gb_params = {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'objective': 'multiclass',
                    'num_class': self.n_policies,
                    'metric': 'multi_logloss',
                    'verbose': -1
                }
        self.gb_params = gb_params
        
        self.transformer_model = None
        self.gb_model = None
        self.feature_dim = None
        self.training_history = {}
        
    def _create_transformer_encoder(self, input_dim: int):
        """Create Transformer encoder for feature extraction."""
        return TransformerFeatureExtractor(
            input_dim=input_dim,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
    
    def fit(
        self,
        patient_features: np.ndarray,
        policy_rewards: pd.DataFrame,
        best_policy: pd.Series,
        val_split: float = 0.1
    ):
        """
        Train hybrid Transformer + GB model.
        
        Phase 1: Train Transformer encoder
        Phase 2: Extract embeddings and train GB classifier
        """
        start_time = time.time()
        
        # Prepare input features
        if self.use_patient_features:
            X = np.hstack([patient_features, policy_rewards[self.policy_names].values])
        else:
            X = policy_rewards[self.policy_names].values
        
        y = best_policy.values
        self.feature_dim = X.shape[1]
        
        # Split data
        n_samples = len(X)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        
        X_train, y_train = X[indices[n_val:]], y[indices[n_val:]]
        X_val, y_val = X[indices[:n_val]], y[indices[:n_val]]
        
        if self.verbose:
            print(f"[Transformer-GB] Phase 1: Training Transformer encoder...")
        
        # Phase 1: Train Transformer encoder
        self.transformer_model = self._create_transformer_encoder(self.feature_dim)
        self._train_transformer_encoder(X_train, y_train, X_val, y_val)
        
        if self.verbose:
            print(f"[Transformer-GB] Phase 2: Extracting embeddings and training GB classifier...")
        
        # Phase 2: Extract embeddings
        train_embeddings = self._extract_embeddings(X_train)
        val_embeddings = self._extract_embeddings(X_val)
        
        # Train GB classifier
        if self.gb_backend == "xgboost":
            self.gb_model = xgb.XGBClassifier(**self.gb_params)
            self.gb_model.fit(
                train_embeddings, y_train,
                eval_set=[(val_embeddings, y_val)],
                verbose=self.verbose
            )
        else:  # lightgbm
            self.gb_model = lgb.LGBMClassifier(**self.gb_params)
            self.gb_model.fit(
                train_embeddings, y_train,
                eval_set=[(val_embeddings, y_val)],
                eval_metric='multi_logloss'
            )
        
        training_time = time.time() - start_time
        
        # Evaluate
        val_preds = self.gb_model.predict(val_embeddings)
        val_acc = np.mean(val_preds == y_val)
        
        if self.verbose:
            print(f"[Transformer-GB] Training complete in {training_time:.1f}s. Val Acc: {val_acc:.4f}")
        
        return self
    
    def _train_transformer_encoder(self, X_train, y_train, X_val, y_val):
        """Train transformer encoder with early stopping."""
        train_dataset = PolicySelectionDataset(X_train, y_train)
        val_dataset = PolicySelectionDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        # For encoder training, we use a temporary classification head
        temp_classifier = nn.Linear(self.hidden_size, self.n_policies).to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer for both encoder and temp classifier
        optimizer = optim.Adam(
            list(self.transformer_model.parameters()) + list(temp_classifier.parameters()),
            lr=self.learning_rate
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
        
        best_val_loss = np.inf
        patience_counter = 0
        best_encoder_state = None
        
        for epoch in range(self.num_epochs):
            # Train
            self.transformer_model.train()
            temp_classifier.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                embeddings = self.transformer_model(batch_X)
                outputs = temp_classifier(embeddings)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.transformer_model.parameters(), self.grad_clip)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validate
            self.transformer_model.eval()
            temp_classifier.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    embeddings = self.transformer_model(batch_X)
                    outputs = temp_classifier(embeddings)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss - self.min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_encoder_state = self.transformer_model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"[Transformer-GB] Encoder early stopping at epoch {epoch+1}")
                break
        
        # Load best encoder
        if best_encoder_state is not None:
            self.transformer_model.load_state_dict(best_encoder_state)
    
    def _extract_embeddings(self, X):
        """Extract embeddings from trained transformer encoder."""
        self.transformer_model.eval()
        dataset = PolicySelectionDataset(X, np.zeros(len(X)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        embeddings = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                batch_embeddings = self.transformer_model(batch_X)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def select(self, patient_features: np.ndarray, policy_rewards_row: np.ndarray) -> str:
        """Select best policy."""
        embedding = self._get_embedding(patient_features, policy_rewards_row)
        idx = int(self.gb_model.predict(embedding)[0])
        return self.policy_names[idx]
    
    def get_policy_probabilities(self, patient_features: np.ndarray, policy_rewards_row: np.ndarray) -> Dict[str, float]:
        """Get probability distribution over policies."""
        embedding = self._get_embedding(patient_features, policy_rewards_row)
        probs = self.gb_model.predict_proba(embedding)[0]
        return {self.policy_names[i]: float(probs[i]) for i in range(len(self.policy_names))}
    
    def _get_embedding(self, patient_features, policy_rewards_row):
        """Get embedding for a single sample."""
        if patient_features.ndim == 1:
            patient_features = patient_features.reshape(1, -1)
        if policy_rewards_row.ndim == 1:
            policy_rewards_row = policy_rewards_row.reshape(1, -1)
        
        if self.use_patient_features:
            X = np.hstack([patient_features, policy_rewards_row])
        else:
            X = policy_rewards_row
        
        self.transformer_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            embedding = self.transformer_model(X_tensor).cpu().numpy()
        
        return embedding


class TransformerFeatureExtractor(nn.Module):
    """Transformer encoder for feature extraction (no classification head)."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.25
    ):
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Extract embeddings."""
        x = x.unsqueeze(1)  # Add sequence dim
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool
        return x


class PolicySelectionDataset(Dataset):
    """PyTorch dataset."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

