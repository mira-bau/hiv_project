from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class RuleBasedPolicy:
    def __init__(self, name: str = "rule_based"):
        self.name = name

    def fit(self, X, y=None):
        return self

    def recommend(self, patient_state: Dict[str, Any]) -> Any:
        return 0


class SupervisedPolicy:
    def __init__(self, model):
        self.model = model
        self.name = getattr(model, "__class__", type("obj", (), {})).__name__

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def recommend(self, patient_state_vec: np.ndarray) -> Any:
        return self.model.predict(patient_state_vec.reshape(1, -1))[0]


class PerActionSupervisedPolicy:
    def __init__(self, base_estimator, actions: List[int], verbose: bool = False):
        self.base_estimator = base_estimator
        self.actions = actions
        self.models: Dict[int, Any] = {}
        self.verbose = verbose

    def fit(self, X: np.ndarray, action: np.ndarray, reward: np.ndarray):
        if self.verbose:
            print(f"[PerAction] Training models for {len(self.actions)} actions...")
        for idx, a in enumerate(self.actions, start=1):
            mask = (action == a)
            if mask.sum() == 0:
                if self.verbose:
                    print(f"[PerAction] Action {a}: no samples, skipping")
                continue
            model = self._clone_estimator()
            model.fit(X[mask], reward[mask])
            self.models[a] = model
            if self.verbose:
                print(f"[PerAction] Trained action {a} ({idx}/{len(self.actions)})")
        if self.verbose:
            print("[PerAction] Done.")
        return self

    def recommend(self, patient_state_vec: np.ndarray) -> Any:
        preds = []
        for a in self.actions:
            model = self.models.get(a)
            if model is None:
                preds.append((-1e9, a))
            else:
                preds.append((float(model.predict(patient_state_vec.reshape(1, -1))[0]), a))
        if not preds:
            return 0
        preds.sort(reverse=True)
        return preds[0][1]

    def _clone_estimator(self):
        cls = self.base_estimator.__class__
        return cls(**getattr(self.base_estimator, 'get_params', lambda: {})())


class DQNPolicy:
    """
    Simple Deep Q-Learning inspired policy using tabular Q-learning with function approximation.
    Uses fitted Q-iteration on the offline dataset.
    """
    def __init__(self, actions: List[int], learning_rate: float = 0.1, gamma: float = 0.9, verbose: bool = False):
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.verbose = verbose
        self.q_models: Dict[int, Any] = {}
        
    def fit(self, X: np.ndarray, action: np.ndarray, reward: np.ndarray, next_X: np.ndarray = None):
        """
        Fit Q-function approximators using offline data.
        Uses a simplified fitted Q-iteration approach.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        if self.verbose:
            print(f"[DQN] Training Q-models for {len(self.actions)} actions...")
        
        # If no next states provided, use simplified approach (reward-only)
        if next_X is None:
            for idx, a in enumerate(self.actions, start=1):
                mask = (action == a)
                if mask.sum() == 0:
                    if self.verbose:
                        print(f"[DQN] Action {a}: no samples, skipping")
                    continue
                # Train Q(s,a) ≈ r for simplicity
                model = GradientBoostingRegressor(n_estimators=50, learning_rate=self.learning_rate, random_state=42)
                model.fit(X[mask], reward[mask])
                self.q_models[a] = model
                if self.verbose:
                    print(f"[DQN] Trained Q-model for action {a} ({idx}/{len(self.actions)})")
        
        if self.verbose:
            print("[DQN] Done.")
        return self
    
    def recommend(self, patient_state_vec: np.ndarray) -> Any:
        """Select action with highest Q-value."""
        q_values = []
        for a in self.actions:
            model = self.q_models.get(a)
            if model is None:
                q_values.append((-1e9, a))
            else:
                q_val = float(model.predict(patient_state_vec.reshape(1, -1))[0])
                q_values.append((q_val, a))
        
        if not q_values:
            return 0
        q_values.sort(reverse=True)
        return q_values[0][1]


class SafetyAwarePolicy:
    """
    Conservative policy that balances reward maximization with safety constraints.
    Avoids actions with high predicted risk (e.g., high variance in outcomes).
    """
    def __init__(self, base_estimator, actions: List[int], safety_threshold: float = 0.3, verbose: bool = False):
        self.base_estimator = base_estimator
        self.actions = actions
        self.safety_threshold = safety_threshold
        self.verbose = verbose
        self.reward_models: Dict[int, Any] = {}
        self.risk_models: Dict[int, Any] = {}
        
    def fit(self, X: np.ndarray, action: np.ndarray, reward: np.ndarray):
        """
        Fit both reward and risk models for each action.
        Risk is estimated as variance/uncertainty in outcomes.
        """
        from sklearn.ensemble import RandomForestRegressor
        
        if self.verbose:
            print(f"[Safety] Training safe models for {len(self.actions)} actions...")
        
        for idx, a in enumerate(self.actions, start=1):
            mask = (action == a)
            if mask.sum() < 10:  # Need enough samples for variance estimation
                if self.verbose:
                    print(f"[Safety] Action {a}: insufficient samples ({mask.sum()}), skipping")
                continue
            
            # Reward model
            reward_model = self._clone_estimator()
            reward_model.fit(X[mask], reward[mask])
            self.reward_models[a] = reward_model
            
            # Risk model: predict variance using Random Forest
            # Use residuals to estimate uncertainty
            preds = reward_model.predict(X[mask])
            residuals = np.abs(reward[mask] - preds)
            risk_model = RandomForestRegressor(n_estimators=30, random_state=42)
            risk_model.fit(X[mask], residuals)
            self.risk_models[a] = risk_model
            
            if self.verbose:
                print(f"[Safety] Trained action {a} ({idx}/{len(self.actions)})")
        
        if self.verbose:
            print("[Safety] Done.")
        return self
    
    def recommend(self, patient_state_vec: np.ndarray) -> Any:
        """
        Select action with best risk-adjusted reward.
        Penalize actions with high predicted risk.
        """
        scores = []
        for a in self.actions:
            reward_model = self.reward_models.get(a)
            risk_model = self.risk_models.get(a)
            
            if reward_model is None or risk_model is None:
                scores.append((-1e9, a))
                continue
            
            predicted_reward = float(reward_model.predict(patient_state_vec.reshape(1, -1))[0])
            predicted_risk = float(risk_model.predict(patient_state_vec.reshape(1, -1))[0])
            
            # Risk-adjusted score: reward - safety_threshold * risk
            safe_score = predicted_reward - self.safety_threshold * predicted_risk
            scores.append((safe_score, a))
        
        if not scores:
            return 0
        scores.sort(reverse=True)
        return scores[0][1]
    
    def _clone_estimator(self):
        cls = self.base_estimator.__class__
        return cls(**getattr(self.base_estimator, 'get_params', lambda: {})())


class RLPolicy:
    def __init__(self, name: str = "rl_policy"):
        self.name = name

    def fit(self, trajectories):
        return self

    def recommend(self, patient_state: Dict[str, Any]) -> Any:
        return 0


class CFKNNPolicy:
    """
    Collaborative Filtering k-Nearest Neighbors policy.
    Uses case-based reasoning by finding similar patients.
    Helps with cold-start and sparse histories.
    """
    def __init__(self, actions: List[int], n_neighbors: int = 10, verbose: bool = False):
        self.actions = actions
        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.knn_models: Dict[int, Any] = {}
        self.scaler = None
        self.feature_indices = None
        
        # Store training data for confidence calculation
        self.training_cf_X: Dict[int, np.ndarray] = {}
        self.training_rewards: Dict[int, np.ndarray] = {}
        
        # Store for meta-features
        self.last_predicted_reward = None
        self.last_neighbor_density = None
        self.last_confidence = None
        
    def _build_cf_features(self, X: np.ndarray, df: pd.DataFrame = None):
        """
        Build compact feature vector for kNN:
        - Recent VL/CD4 stats (last value + trend over N visits)
        - Current regimen one-hots
        - Gender, ethnicity
        """
        from sklearn.preprocessing import StandardScaler
        
        # If df is provided, extract structured features
        if df is not None:
            # Use a subset: VL, CD4, Gender, Ethnic, regimen features
            # For simplicity, use first N features from X (assuming they're ordered)
            # This assumes VL is first, CD4 is second, etc.
            feature_cols = ['VL', 'CD4', 'Gender', 'Ethnic', 'Base Drug Combo', 
                          'Comp. INI', 'Comp. NNRTI', 'Extra PI', 'Extra pk-En']
            
            # Try to extract from df if columns exist
            cf_features = []
            available_cols = []
            for col in feature_cols:
                if col in df.columns:
                    available_cols.append(col)
            
            if len(available_cols) > 0:
                # Use df-based features
                cf_X = df[available_cols].values
            else:
                # Fallback: use first few features from X (VL, CD4, etc.)
                n_cf_features = min(9, X.shape[1])
                cf_X = X[:, :n_cf_features]
        else:
            # Fallback: use first few features from X
            n_cf_features = min(9, X.shape[1])
            cf_X = X[:, :n_cf_features]
        
        # Standardize features
        if self.scaler is None:
            self.scaler = StandardScaler()
            cf_X_scaled = self.scaler.fit_transform(cf_X)
            self.feature_indices = list(range(cf_X.shape[1]))
        else:
            cf_X_scaled = self.scaler.transform(cf_X)
        
        return cf_X_scaled
    
    def fit(self, X: np.ndarray, action: np.ndarray, reward: np.ndarray, df: pd.DataFrame = None):
        """
        Fit kNN models for each action using collaborative filtering.
        """
        from sklearn.neighbors import KNeighborsRegressor
        
        if self.verbose:
            print(f"[CF-kNN] Training kNN models for {len(self.actions)} actions...")
        
        # Build compact CF features
        cf_X = self._build_cf_features(X, df)
        
        for idx, a in enumerate(self.actions, start=1):
            mask = (action == a)
            if mask.sum() == 0:
                if self.verbose:
                    print(f"[CF-kNN] Action {a}: no samples, skipping")
                continue
            
            # Train kNN regressor for this action
            n_neighbors_actual = min(self.n_neighbors, mask.sum())
            if n_neighbors_actual < 1:
                continue
                
            model = KNeighborsRegressor(
                n_neighbors=n_neighbors_actual,
                weights='distance',
                algorithm='auto',
                metric='euclidean'
            )
            model.fit(cf_X[mask], reward[mask])
            self.knn_models[a] = model
            # Store training data for confidence calculation
            self.training_cf_X[a] = cf_X[mask]
            self.training_rewards[a] = reward[mask]
            
            if self.verbose:
                print(f"[CF-kNN] Trained action {a} ({idx}/{len(self.actions)}), neighbors={n_neighbors_actual})")
        
        if self.verbose:
            print("[CF-kNN] Done.")
        return self
    
    def recommend(self, patient_state_vec: np.ndarray, df_row: pd.DataFrame = None) -> Any:
        """
        Select action with highest predicted reward from kNN.
        Also computes meta-features for the meta-selector.
        """
        from sklearn.neighbors import KNeighborsRegressor
        
        if not self.knn_models:
            return 0
        
        # Build CF features for this patient
        cf_X = self._build_cf_features(
            patient_state_vec.reshape(1, -1), 
            df_row if df_row is not None else None
        )
        
        # Predict reward for each action
        predicted_rewards = []
        best_action_neighbor_rewards = []
        
        for a in self.actions:
            model = self.knn_models.get(a)
            if model is None:
                predicted_rewards.append((-1e9, a))
                continue
            
            # Get prediction
            pred_reward = float(model.predict(cf_X)[0])
            predicted_rewards.append((pred_reward, a))
        
        if not predicted_rewards:
            self.last_predicted_reward = 0.0
            self.last_neighbor_density = 0
            self.last_confidence = 0.0
            return 0
        
        # Select best action
        predicted_rewards.sort(reverse=True)
        best_reward, best_action = predicted_rewards[0]
        
        # Get neighbor rewards for best action for confidence calculation
        model_best = self.knn_models.get(best_action)
        if model_best is not None and best_action in self.training_rewards:
            distances, indices = model_best.kneighbors(cf_X, return_distance=True)
            if len(indices[0]) > 0:
                # Get actual neighbor rewards from training data
                neighbor_indices = indices[0]
                best_action_neighbor_rewards = self.training_rewards[best_action][neighbor_indices].tolist()
        
        # Compute meta-features
        self.last_predicted_reward = best_reward
        
        # Neighbor density: number of valid neighbors
        if model_best is not None:
            _, indices = model_best.kneighbors(cf_X)
            self.last_neighbor_density = len(indices[0])
        else:
            self.last_neighbor_density = 0
        
        # Confidence: 1 / (1 + variance of neighbor rewards)
        if len(best_action_neighbor_rewards) > 1:
            neighbor_var = np.var(best_action_neighbor_rewards)
            self.last_confidence = 1.0 / (1.0 + neighbor_var)
        else:
            self.last_confidence = 0.5  # Default moderate confidence
        
        return best_action
    
    def get_meta_features(self) -> Dict[str, float]:
        """Return meta-features for the meta-selector."""
        return {
            'cf_predicted_reward': self.last_predicted_reward if self.last_predicted_reward is not None else 0.0,
            'cf_neighbor_density': self.last_neighbor_density if self.last_neighbor_density is not None else 0,
            'cf_confidence': self.last_confidence if self.last_confidence is not None else 0.0
        }
