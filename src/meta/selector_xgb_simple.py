"""
XGBoost Simple Meta-Selector for Policy Selection

This module implements a simple XGBoost-based meta-selector that uses only patient state features
(no policy or CF features) as a clean baseline.

The architecture:
- State features only: (n_samples, state_dim)
- XGBoost multi-class classifier with class weights (via sample_weight)
- Predicts policy indices (0..4) directly
"""

from typing import List, Optional
import numpy as np
import time

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


class XGBoostSimpleMetaSelector:
    """
    Simple XGBoost meta-selector using only patient state features.
    
    Uses state feature vectors:
    - State features: (n_samples, state_dim)
    """
    
    def __init__(
        self,
        policy_names: List[str],
        max_depth: int = 4,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        use_class_weights: bool = True,
        verbose: bool = True,
        random_state: int = 42
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        self.policy_names = policy_names
        self.n_policies = len(policy_names)
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.use_class_weights = use_class_weights
        self.verbose = verbose
        self.random_state = random_state
        
        self.model = None
        self.policy_to_idx = {name: idx for idx, name in enumerate(policy_names)}
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'XGBoostSimpleMetaSelector':
        """
        Train XGBoost classifier on state features.
        
        Args:
            X_train: State feature vectors (n_samples, state_dim)
            y_train: Policy indices (n_samples,) with values in [0, n_policies-1]
            
        Returns:
            self
        """
        start_time = time.time()
        
        # Compute class weights for imbalanced classes
        sample_weight = None
        if self.use_class_weights:
            class_counts = np.bincount(y_train, minlength=self.n_policies)
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.mean()  # Normalize by mean
            sample_weight = np.array([class_weights[int(y)] for y in y_train])
            
            if self.verbose:
                print(f"[XGB SIMPLE] Train label distribution (indices): {class_counts.tolist()}")
                label_dist_names = {self.policy_names[i] if i < len(self.policy_names) else f"unknown_{i}": int(count) 
                                   for i, count in enumerate(class_counts)}
                print(f"[XGB SIMPLE] Train label distribution (names): {label_dist_names}")
                print(f"[XGB SIMPLE] Class weights: {class_weights.tolist()}")
        
        # Create and train XGBoost classifier
        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            objective='multi:softprob',  # Multi-class classification with probabilities
            num_class=self.n_policies,
            random_state=self.random_state,
            verbosity=1 if self.verbose else 0
        )
        
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        if self.verbose:
            print(f"[XGB SIMPLE] Training complete in {elapsed:.1f}s")
        
        return self
    
    def select(self, X: np.ndarray) -> np.ndarray:
        """
        Predict policy indices for given state feature vectors.
        
        Args:
            X: State feature vectors (n_samples, state_dim)
            
        Returns:
            Policy indices (n_samples,) as np.ndarray
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        predictions = self.model.predict(X)
        return predictions.astype(int)
    
    def get_policy_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability distribution over policies for given state features.
        
        Args:
            X: State feature vectors (n_samples, state_dim)
            
        Returns:
            Probability matrix (n_samples, n_policies)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.model.predict_proba(X)


