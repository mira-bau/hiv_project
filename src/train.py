import os
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor
import mlflow

from src.data.loaders import load_sample_dataset, load_csv_with_limit
from src.features.state import build_state_action_table, BASE_FEATURES, ACTION_COL, OUTCOME_COLS
from src.features.sequences import build_patient_sequences
from src.models.baselines import RuleBasedPolicy, PerActionSupervisedPolicy, DQNPolicy, SafetyAwarePolicy, CFKNNPolicy
from src.eval.ips import estimate_ips, estimate_snips, estimate_dr

# Try to import TransformerGB selector
try:
    from src.meta.selector_transformer_gb import TransformerGBMetaSelector
    TRANSFORMER_GB_AVAILABLE = True
except ImportError:
    TRANSFORMER_GB_AVAILABLE = False
    print("Warning: TransformerGBMetaSelector not available. Install PyTorch to use transformer_gb mode.")

# Try to import StaticMultiContext selector
try:
    from src.meta.selector_static_multi_context import StaticMultiContextMetaSelector
    STATIC_MULTI_CONTEXT_AVAILABLE = True
except ImportError:
    STATIC_MULTI_CONTEXT_AVAILABLE = False
    print("Warning: StaticMultiContextMetaSelector not available. Install PyTorch to use static_multi_context mode.")

# Try to import XGBoostSimple selector
try:
    from src.meta.selector_xgb_simple import XGBoostSimpleMetaSelector
    XGBOOST_SIMPLE_AVAILABLE = True
except ImportError:
    XGBOOST_SIMPLE_AVAILABLE = False
    print("Warning: XGBoostSimpleMetaSelector not available. Install xgboost to use xgboost_simple mode.")


def make_reward(vl: np.ndarray, cd4: np.ndarray) -> np.ndarray:
    if len(vl) == 0:
        return np.array([])
    vl_norm = (vl - vl.mean()) / (vl.std() + 1e-8)
    cd4_norm = (cd4 - cd4.mean()) / (cd4.std() + 1e-8)
    return (-vl_norm + cd4_norm)


def patient_split(df: pd.DataFrame, val_frac: float, seed: int = 42):
    pids = df["PatientID"].unique()
    if len(pids) <= 1:
        return df.copy(), df.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n_val = max(1, int(len(pids) * val_frac))
    if n_val >= len(pids):
        n_val = 1
    val_ids = set(pids[:n_val])
    train_df = df[~df["PatientID"].isin(val_ids)].copy()
    val_df = df[df["PatientID"].isin(val_ids)].copy()
    if len(train_df) == 0:
        train_df = df.copy()
        val_df = df.copy()
    return train_df, val_df


def compute_policy_actions(policies_dict, X, df=None):
    """Compute actions for all enabled policies."""
    actions = {}
    cf_meta_features = {}
    
    for name, policy in policies_dict.items():
        if policy is None:
            continue
        if name == "rule":
            actions[name] = np.array([policy.recommend({}) for _ in range(len(X))])
        elif name == "cf_knn":
            # CF-kNN needs special handling for meta-features
            cf_actions = []
            cf_pred_rewards = []
            cf_neighbor_densities = []
            cf_confidences = []
            
            for i in range(len(X)):
                df_row = df.iloc[[i]] if df is not None else None
                action = policy.recommend(X[i], df_row)
                cf_actions.append(action)
                
                # Extract meta-features
                meta = policy.get_meta_features()
                cf_pred_rewards.append(meta['cf_predicted_reward'])
                cf_neighbor_densities.append(meta['cf_neighbor_density'])
                cf_confidences.append(meta['cf_confidence'])
            
            actions[name] = np.array(cf_actions)
            cf_meta_features['cf_predicted_reward'] = np.array(cf_pred_rewards)
            cf_meta_features['cf_neighbor_density'] = np.array(cf_neighbor_densities)
            cf_meta_features['cf_confidence'] = np.array(cf_confidences)
        else:
            actions[name] = np.array([policy.recommend(X[i]) for i in range(len(X))])
    
    return actions, cf_meta_features


def build_static_multi_context_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    policies: dict,
    feature_cols: list,
    cfg,
    policy_rewards_tr: pd.DataFrame,
    policy_rewards_va: pd.DataFrame,
    cf_meta_features_tr: dict,
    cf_meta_features_va: dict,
    policy_names: list
) -> tuple:
    """
    Build static multi-context features for StaticMultiContextMetaSelector.
    
    Returns:
        (state_x_tr, policy_feats_tr, neighbor_feats_tr, state_x_va, policy_feats_va, neighbor_feats_va)
    """
    # 1. Build state features: (n_samples, state_dim)
    state_x_tr = train_df[feature_cols].values
    state_x_va = val_df[feature_cols].values
    state_dim = state_x_tr.shape[1]
    print(f"[STATIC META] State features: train={state_x_tr.shape}, val={state_x_va.shape}, state_dim={state_dim}")
    
    # 2. Build policy features: (n_samples, num_policies, policy_feat_dim)
    # Each policy gets [predicted_reward] (minimum, can add more features later)
    n_policies = len(policy_names)
    policy_feat_dim = 1  # Start with just predicted_reward
    
    policy_feats_tr = np.zeros((len(train_df), n_policies, policy_feat_dim))
    for j, policy_name in enumerate(policy_names):
        policy_feats_tr[:, j, 0] = policy_rewards_tr[policy_name].values  # predicted_reward
    
    policy_feats_va = np.zeros((len(val_df), n_policies, policy_feat_dim))
    for j, policy_name in enumerate(policy_names):
        policy_feats_va[:, j, 0] = policy_rewards_va[policy_name].values  # predicted_reward
    
    # Apply per-sample z-score normalization to predicted_reward if configured
    reward_normalization_method = cfg.meta.get("static_reward_normalization", "zscore")
    if reward_normalization_method != "none":
        print(f"[STATIC META] Using reward normalization: {reward_normalization_method}")
        for i in range(len(policy_feats_tr)):
            r_i = policy_feats_tr[i, :, 0]
            if reward_normalization_method == "zscore":
                mean_i = np.mean(r_i)
                std_i = np.std(r_i)
                if std_i < 1e-6:
                    std_i = 1e-6
                policy_feats_tr[i, :, 0] = (r_i - mean_i) / std_i
            elif reward_normalization_method == "relative":
                policy_feats_tr[i, :, 0] = r_i - np.mean(r_i)
        
        for i in range(len(policy_feats_va)):
            r_i = policy_feats_va[i, :, 0]
            if reward_normalization_method == "zscore":
                mean_i = np.mean(r_i)
                std_i = np.std(r_i)
                if std_i < 1e-6:
                    std_i = 1e-6
                policy_feats_va[i, :, 0] = (r_i - mean_i) / std_i
            elif reward_normalization_method == "relative":
                policy_feats_va[i, :, 0] = r_i - np.mean(r_i)
    
    print(f"[STATIC META] Policy features: train={policy_feats_tr.shape}, val={policy_feats_va.shape}, feat_dim={policy_feat_dim}")
    
    # 3. Build neighbor features: (n_samples, neighbor_dim=3)
    # [cf_predicted_reward, cf_neighbor_density, cf_confidence]
    neighbor_feats_tr = np.zeros((len(train_df), 3))
    if cf_meta_features_tr and len(cf_meta_features_tr) > 0:
        neighbor_feats_tr[:, 0] = cf_meta_features_tr.get('cf_predicted_reward', np.zeros(len(train_df)))
        neighbor_feats_tr[:, 1] = cf_meta_features_tr.get('cf_neighbor_density', np.zeros(len(train_df)))
        neighbor_feats_tr[:, 2] = cf_meta_features_tr.get('cf_confidence', np.full(len(train_df), 0.5))
    else:
        # Default values if CF features not available
        neighbor_feats_tr[:, 0] = 0.0
        neighbor_feats_tr[:, 1] = 0.0
        neighbor_feats_tr[:, 2] = 0.5
    
    neighbor_feats_va = np.zeros((len(val_df), 3))
    if cf_meta_features_va and len(cf_meta_features_va) > 0:
        neighbor_feats_va[:, 0] = cf_meta_features_va.get('cf_predicted_reward', np.zeros(len(val_df)))
        neighbor_feats_va[:, 1] = cf_meta_features_va.get('cf_neighbor_density', np.zeros(len(val_df)))
        neighbor_feats_va[:, 2] = cf_meta_features_va.get('cf_confidence', np.full(len(val_df), 0.5))
    else:
        # Default values if CF features not available
        neighbor_feats_va[:, 0] = 0.0
        neighbor_feats_va[:, 1] = 0.0
        neighbor_feats_va[:, 2] = 0.5
    
    print(f"[STATIC META] Neighbor features: train={neighbor_feats_tr.shape}, val={neighbor_feats_va.shape}")
    
    return (
        state_x_tr, policy_feats_tr, neighbor_feats_tr,
        state_x_va, policy_feats_va, neighbor_feats_va
    )


def train_meta_selector(train_df, val_df, policies, feature_cols, cfg):
    """Train meta-selector to choose best policy per patient-time."""
    print("[Stage] Training meta-selector...")
    
    def to_arrays(d: pd.DataFrame):
        if len(d) == 0:
            return np.empty((0, len(feature_cols))), np.array([]), np.array([])
        X = d[feature_cols].values
        a = d[cfg.columns.action].values.astype(int)
        vl = d[cfg.columns.vl_out].values
        cd4 = d[cfg.columns.cd4_out].values
        r = make_reward(vl, cd4)
        return X, a, r

    Xtr, atr, rtr = to_arrays(train_df)
    Xva, ava, rva = to_arrays(val_df)

    # Generate policy actions for training data
    train_actions, cf_meta_features_tr = compute_policy_actions(policies, Xtr, train_df)
    
    # Add CF-kNN meta-features to patient features if available
    if cf_meta_features_tr and len(cf_meta_features_tr) > 0:
        # Append cf meta-features as additional columns to Xtr
        cf_features_array = np.column_stack([
            cf_meta_features_tr.get('cf_predicted_reward', np.zeros(len(Xtr))),
            cf_meta_features_tr.get('cf_neighbor_density', np.zeros(len(Xtr))),
            cf_meta_features_tr.get('cf_confidence', np.zeros(len(Xtr)))
        ])
        Xtr = np.hstack([Xtr, cf_features_array])
        print(f"[Info] Added CF-kNN meta-features to training features. New shape: {Xtr.shape}")
    
    # Create policy reward predictions (simplified: use actual rewards where actions match)
    policy_rewards_tr = pd.DataFrame(index=range(len(Xtr)))
    policy_names = []
    
    for name, actions in train_actions.items():
        policy_names.append(name)
        # For each policy, predict reward based on whether its action matches logged action
        rewards = np.zeros(len(Xtr))
        for i in range(len(Xtr)):
            if actions[i] == atr[i]:  # Policy action matches logged action
                rewards[i] = rtr[i]  # Use actual reward
            else:
                # Use average reward as fallback
                rewards[i] = np.mean(rtr)
        policy_rewards_tr[name] = rewards

    # Determine best policy per sample (highest predicted reward)
    best_policy_idx = policy_rewards_tr.values.argmax(axis=1)
    
    # Train meta-selector based on type
    selector_type = cfg.meta.get("selector_type", "rf")
    
    # Handle static_multi_context selector (no temporal sequences)
    if selector_type == "static_multi_context":
        if not STATIC_MULTI_CONTEXT_AVAILABLE:
            print("[Warning] StaticMultiContextMetaSelector not available, falling back to Random Forest")
            selector_type = "rf"
        else:
            # Build validation policy rewards first (needed for feature building)
            val_actions_temp, cf_meta_features_va_temp = compute_policy_actions(policies, Xva, val_df)
            policy_rewards_va_temp = pd.DataFrame(index=range(len(Xva)))
            for name, actions in val_actions_temp.items():
                rewards = np.zeros(len(Xva))
                for i in range(len(Xva)):
                    if actions[i] == ava[i]:
                        rewards[i] = rva[i]
                    else:
                        rewards[i] = np.mean(rva)
                policy_rewards_va_temp[name] = rewards
            
            # Build static multi-context features
            (state_x_tr, policy_feats_tr, neighbor_feats_tr,
             state_x_va, policy_feats_va, neighbor_feats_va) = build_static_multi_context_features(
                train_df, val_df, policies, feature_cols, cfg,
                policy_rewards_tr, policy_rewards_va_temp,
                cf_meta_features_tr, cf_meta_features_va_temp,
                policy_names
            )
            
            # Compute best_policy_idx using normalized rewards per-sample
            # Normalize rewards per-sample to give all policies a fair chance
            policy_rewards_array = policy_rewards_tr.values  # (n_samples, n_policies)
            
            # Per-sample normalization: z-score normalize across policies for each sample
            normalized_rewards = np.zeros_like(policy_rewards_array)
            for i in range(len(policy_rewards_array)):
                r_i = policy_rewards_array[i]
                mean_i = np.mean(r_i)
                std_i = np.std(r_i)
                if std_i < 1e-6:
                    std_i = 1e-6
                normalized_rewards[i] = (r_i - mean_i) / std_i
            
            # Use normalized rewards to select best policy
            best_policy_idx = normalized_rewards.argmax(axis=1)
            
            # Debug logging
            if cfg.meta.get("selector_type") == "static_multi_context":
                unique, counts = np.unique(best_policy_idx, return_counts=True)
                label_dist = {policy_names[int(idx)] if int(idx) < len(policy_names) else f"unknown_{idx}": int(count) 
                             for idx, count in zip(unique, counts)}
                print(f"[STATIC META] train label dist: {label_dist}")
            
            # Train StaticMultiContextMetaSelector
            meta_selector = StaticMultiContextMetaSelector(
                policy_names=policy_names,
                state_dim=state_x_tr.shape[1],
                policy_feat_dim=policy_feats_tr.shape[2],
                neighbor_dim=neighbor_feats_tr.shape[1],
                d_model=cfg.meta.get("static_d_model", 128),
                n_heads=cfg.meta.get("static_nheads", 4),
                n_layers=cfg.meta.get("static_nlayers", 2),
                dim_feedforward=cfg.meta.get("static_dim_feedforward", 256),
                dropout=cfg.meta.get("static_dropout", 0.15),
                learning_rate=cfg.meta.get("static_lr", 0.001),
                batch_size=cfg.meta.get("static_batch_size", 64),
                num_epochs=cfg.meta.get("static_epochs", 25),
                patience=cfg.meta.get("static_patience", 5),
                min_delta=cfg.meta.get("static_min_delta", 1e-4),
                use_class_weights=cfg.meta.get("static_use_class_weights", True),
                device=cfg.meta.get("device", "cpu"),
                verbose=True
            ).fit(
                state_x_tr, policy_feats_tr, neighbor_feats_tr,
                best_policy_idx, val_split=0.1
            )
            
            # Store features for evaluation
            meta_selector._state_x_va = state_x_va
            meta_selector._policy_feats_va = policy_feats_va
            meta_selector._neighbor_feats_va = neighbor_feats_va
            meta_selector._val_actions = val_actions_temp
            meta_selector._policy_rewards_va = policy_rewards_va_temp
    
    # Handle xgboost_simple selector (pure XGBoost with state features only)
    elif selector_type == "xgboost_simple":
        if not XGBOOST_SIMPLE_AVAILABLE:
            print("[Warning] XGBoostSimpleMetaSelector not available, falling back to Random Forest")
            selector_type = "rf"
        else:
            # Build validation policy rewards first (needed for label generation)
            val_actions_temp, cf_meta_features_va_temp = compute_policy_actions(policies, Xva, val_df)
            policy_rewards_va_temp = pd.DataFrame(index=range(len(Xva)))
            for name, actions in val_actions_temp.items():
                rewards = np.zeros(len(Xva))
                for i in range(len(Xva)):
                    if actions[i] == ava[i]:
                        rewards[i] = rva[i]
                    else:
                        rewards[i] = np.mean(rva)
                policy_rewards_va_temp[name] = rewards
            
            # Build state features only (no policy or CF features)
            X_train_state = train_df[feature_cols].values
            X_val_state = val_df[feature_cols].values
            
            print(f"[XGB SIMPLE] X_train_state shape: {X_train_state.shape}")
            print(f"[XGB SIMPLE] X_val_state shape: {X_val_state.shape}")
            
            # Compute best_policy_idx using normalized rewards per-sample (same as static_multi_context)
            policy_rewards_array = policy_rewards_tr.values  # (n_samples, n_policies)
            
            # Per-sample normalization: z-score normalize across policies for each sample
            normalized_rewards = np.zeros_like(policy_rewards_array)
            for i in range(len(policy_rewards_array)):
                r_i = policy_rewards_array[i]
                mean_i = np.mean(r_i)
                std_i = np.std(r_i)
                if std_i < 1e-6:
                    std_i = 1e-6
                normalized_rewards[i] = (r_i - mean_i) / std_i
            
            # Use normalized rewards to select best policy
            best_policy_idx = normalized_rewards.argmax(axis=1)
            
            # Log label distribution
            unique, counts = np.unique(best_policy_idx, return_counts=True)
            label_dist = {policy_names[int(idx)] if int(idx) < len(policy_names) else f"unknown_{idx}": int(count) 
                         for idx, count in zip(unique, counts)}
            print(f"[XGB SIMPLE] y_train label dist: {label_dist}")
            
            # Train XGBoostSimpleMetaSelector
            meta_selector = XGBoostSimpleMetaSelector(
                policy_names=policy_names,
                max_depth=cfg.meta.get("xgb_simple_max_depth", 4),
                n_estimators=cfg.meta.get("xgb_simple_n_estimators", 200),
                learning_rate=cfg.meta.get("xgb_simple_learning_rate", 0.05),
                subsample=cfg.meta.get("xgb_simple_subsample", 0.8),
                colsample_bytree=cfg.meta.get("xgb_simple_colsample_bytree", 0.8),
                reg_lambda=cfg.meta.get("xgb_simple_reg_lambda", 1.0),
                reg_alpha=cfg.meta.get("xgb_simple_reg_alpha", 0.0),
                use_class_weights=cfg.meta.get("xgb_simple_use_class_weights", True),
                verbose=True,
                random_state=cfg.get("seed", 42)
            ).fit(X_train_state, best_policy_idx)
            
            # Store features for evaluation
            meta_selector._X_val_state = X_val_state
            meta_selector._val_actions = val_actions_temp
            meta_selector._policy_rewards_va = policy_rewards_va_temp
    
    elif selector_type in ["transformer", "transformer_v2", "transformer_gb"]:
        if not TRANSFORMER_AVAILABLE:
            print("[Warning] Transformer not available, falling back to Random Forest")
            selector_type = "rf"
        else:
            if selector_type == "transformer_v2" or selector_type == "transformer":
                # Use improved v2 by default for "transformer"
                meta_selector = TransformerMetaSelectorV2(
                    policy_names=policy_names,
                    use_patient_features=cfg.meta.use_patient_features,
                    hidden_size=cfg.meta.get("transformer_hidden_size", 128),
                    num_heads=cfg.meta.get("transformer_num_heads", 4),
                    num_layers=cfg.meta.get("transformer_num_layers", 2),
                    dropout=cfg.meta.get("transformer_dropout", 0.25),
                    learning_rate=cfg.meta.get("transformer_lr", 0.001),
                    batch_size=cfg.meta.get("transformer_batch_size", 64),
                    num_epochs=cfg.meta.get("transformer_epochs", 50),
                    patience=cfg.meta.get("transformer_patience", 8),
                    min_delta=cfg.meta.get("transformer_min_delta", 1e-4),
                    grad_clip=cfg.meta.get("transformer_grad_clip", 1.0),
                    use_mask=cfg.meta.get("transformer_use_mask", True),
                    lr_scheduler=cfg.meta.get("lr_scheduler", "reduce_on_plateau"),
                    device=cfg.meta.get("device", "cpu"),
                    verbose=True
                ).fit(Xtr, policy_rewards_tr, pd.Series(best_policy_idx), val_split=0.1)
            
            elif selector_type == "transformer_gb":
                meta_selector = TransformerGBMetaSelector(
                    policy_names=policy_names,
                    use_patient_features=cfg.meta.use_patient_features,
                    hidden_size=cfg.meta.get("transformer_hidden_size", 128),
                    num_heads=cfg.meta.get("transformer_num_heads", 4),
                    num_layers=cfg.meta.get("transformer_num_layers", 2),
                    dropout=cfg.meta.get("transformer_dropout", 0.25),
                    learning_rate=cfg.meta.get("transformer_lr", 0.001),
                    batch_size=cfg.meta.get("transformer_batch_size", 64),
                    num_epochs=cfg.meta.get("transformer_epochs", 50),
                    patience=cfg.meta.get("transformer_patience", 8),
                    min_delta=cfg.meta.get("transformer_min_delta", 1e-4),
                    grad_clip=cfg.meta.get("transformer_grad_clip", 1.0),
                    gb_backend=cfg.meta.get("gb_backend", "xgboost"),
                    device=cfg.meta.get("device", "cpu"),
                    verbose=True
                ).fit(Xtr, policy_rewards_tr, pd.Series(best_policy_idx), val_split=0.1)
    
    if selector_type not in ["transformer", "transformer_v2", "transformer_gb", "multi_context", "static_multi_context", "xgboost_simple"]:
        meta_selector = MetaSelector(
            policy_names=policy_names,
            use_patient_features=cfg.meta.use_patient_features,
            model_type=cfg.meta.model_type
        ).fit(Xtr, policy_rewards_tr, pd.Series(best_policy_idx))
    
    print(f"[Info] Meta-selector ({selector_type}) trained on {len(Xtr)} samples with {len(policy_names)} policies")
    
    # Evaluate meta-selector on validation set
    if selector_type == "static_multi_context":
        # Use pre-computed features for static_multi_context
        state_x_va = meta_selector._state_x_va
        policy_feats_va = meta_selector._policy_feats_va
        neighbor_feats_va = meta_selector._neighbor_feats_va
        val_actions = meta_selector._val_actions
        policy_rewards_va = meta_selector._policy_rewards_va
        
        # Meta-selector recommendations for static_multi_context
        meta_selected_policy_indices_va = []
        meta_actions_va = []
        meta_selected_policies_va = []
        meta_rewards = []
        
        for i in range(len(state_x_va)):
            selected_policy_name = meta_selector.select(
                state_x_va[i:i+1],  # (1, state_dim)
                policy_feats_va[i:i+1],  # (1, num_policies, policy_feat_dim)
                neighbor_feats_va[i:i+1]  # (1, neighbor_dim)
            )
            # Map policy name to index
            selected_policy_idx = policy_names.index(selected_policy_name) if selected_policy_name in policy_names else 0
            
            meta_action = val_actions[selected_policy_name][i]
            meta_reward = policy_rewards_va.iloc[i][selected_policy_name]
            
            meta_selected_policy_indices_va.append(selected_policy_idx)
            meta_actions_va.append(meta_action)
            meta_selected_policies_va.append(selected_policy_name)
            meta_rewards.append(meta_reward)
        
        # Convert to numpy arrays
        meta_selected_policy_indices_va = np.array(meta_selected_policy_indices_va)
        meta_actions_va = np.array(meta_actions_va)
        meta_selected_policies_va = np.array(meta_selected_policies_va)
        meta_rewards = np.array(meta_rewards)
        
        # Debug logging: Print real meta-selector predictions
        if cfg.meta.get("selector_type") == "static_multi_context":
            unique_idx, counts = np.unique(meta_selected_policy_indices_va, return_counts=True)
            pred_dist_idx = dict(zip(unique_idx.astype(int), counts.astype(int)))
            pred_dist_names = {policy_names[i] if i < len(policy_names) else f"unknown_{i}": int(c) 
                              for i, c in zip(unique_idx, counts)}
            print(f"[STATIC META] val meta_actions: {pred_dist_names}")
            
            # Debug: Verify val_actions structure
            print("[IPS DEBUG] val_actions keys:", list(val_actions.keys()))
            for name, actions in val_actions.items():
                unique_actions, action_counts = np.unique(actions, return_counts=True)
                print(f"[IPS DEBUG] val_actions[{name}] shape: {actions.shape}, unique: {dict(zip(unique_actions.astype(int), action_counts.astype(int)))}")
            
            # Debug: Check first few meta actions
            print(f"[IPS DEBUG] First 10 meta_actions_va: {meta_actions_va[:10]}")
            print(f"[IPS DEBUG] First 10 selected policies: {meta_selected_policies_va[:10]}")
            for i in range(min(5, len(meta_actions_va))):
                print(f"[IPS DEBUG] sample {i}: policy={meta_selected_policies_va[i]}, meta_action={meta_actions_va[i]}, val_actions[{meta_selected_policies_va[i]}][{i}]={val_actions[meta_selected_policies_va[i]][i]}")
        
        # Direct 1:1 mapping (no sequence alignment needed)
        meta_actions = meta_actions_va
        meta_rewards = np.array(meta_rewards)
        meta_selected_policies = meta_selected_policies_va
    
    elif selector_type == "xgboost_simple":
        # Use pre-computed features for xgboost_simple
        X_val_state = meta_selector._X_val_state
        val_actions = meta_selector._val_actions
        policy_rewards_va = meta_selector._policy_rewards_va
        
        # Meta-selector recommendations for xgboost_simple
        # select() returns policy indices (0..4)
        meta_policy_idx_val = meta_selector.select(X_val_state)
        
        # Convert indices to policy names
        meta_selected_policies_va = np.array([policy_names[int(idx)] if int(idx) < len(policy_names) else policy_names[0] 
                                              for idx in meta_policy_idx_val])
        
        # Get actions from corresponding base policies
        meta_actions_va = []
        meta_rewards = []
        for i in range(len(meta_policy_idx_val)):
            policy_name = meta_selected_policies_va[i]
            meta_action = val_actions[policy_name][i]
            meta_reward = policy_rewards_va.iloc[i][policy_name]
            meta_actions_va.append(meta_action)
            meta_rewards.append(meta_reward)
        
        # Convert to numpy arrays
        meta_actions_va = np.array(meta_actions_va)
        meta_rewards = np.array(meta_rewards)
        
        # Log prediction distribution
        unique_idx, counts = np.unique(meta_policy_idx_val, return_counts=True)
        pred_dist_names = {policy_names[i] if i < len(policy_names) else f"unknown_{i}": int(c) 
                          for i, c in zip(unique_idx, counts)}
        print(f"[XGB SIMPLE] val meta_actions (policy names): {pred_dist_names}")
        
        # Direct 1:1 mapping (no sequence alignment needed)
        meta_actions = meta_actions_va
        meta_selected_policies = meta_selected_policies_va
        
        # Store aligned actions for DR computation
        meta_selector._meta_actions_aligned = meta_actions
    
    else:
        # Standard evaluation for other selectors
        val_actions, cf_meta_features_va = compute_policy_actions(policies, Xva, val_df)
        
        # Add CF-kNN meta-features to validation features if available
        if cf_meta_features_va and len(cf_meta_features_va) > 0:
            cf_features_array_va = np.column_stack([
                cf_meta_features_va.get('cf_predicted_reward', np.zeros(len(Xva))),
                cf_meta_features_va.get('cf_neighbor_density', np.zeros(len(Xva))),
                cf_meta_features_va.get('cf_confidence', np.zeros(len(Xva)))
            ])
            Xva = np.hstack([Xva, cf_features_array_va])
            print(f"[Info] Added CF-kNN meta-features to validation features. New shape: {Xva.shape}")
        
        policy_rewards_va = pd.DataFrame(index=range(len(Xva)))
        
        for name, actions in val_actions.items():
            rewards = np.zeros(len(Xva))
            for i in range(len(Xva)):
                if actions[i] == ava[i]:
                    rewards[i] = rva[i]
                else:
                    rewards[i] = np.mean(rva)
            policy_rewards_va[name] = rewards

        # Meta-selector recommendations
        meta_actions_list = []
        meta_rewards_list = []
        for i in range(len(Xva)):
            selected_policy = meta_selector.select(Xva[i], policy_rewards_va.iloc[i].values)
            meta_action = val_actions[selected_policy][i]
            meta_reward = policy_rewards_va.iloc[i][selected_policy]
            meta_actions_list.append(meta_action)
            meta_rewards_list.append(meta_reward)
        
        meta_actions = np.array(meta_actions_list)
        meta_rewards = np.array(meta_rewards_list)
    
    # Store aligned actions in selector for DR computation in main()
    meta_selector._meta_actions_aligned = meta_actions
    
    # Debug prints before IPS computation (only for static_multi_context)
    if selector_type == "static_multi_context":
        print("[IPS DEBUG] meta actions shape:", meta_actions.shape)
        unique_meta, counts_meta = np.unique(meta_actions, return_counts=True)
        print("[IPS DEBUG] meta actions unique values:", dict(zip(unique_meta.astype(int), counts_meta.astype(int))))
        print("[IPS DEBUG] example meta actions (first 20):", meta_actions[:20].tolist())
        unique_ava, counts_ava = np.unique(ava, return_counts=True)
        print("[IPS DEBUG] logged actions (ava) unique values:", dict(zip(unique_ava.astype(int), counts_ava.astype(int))))
        print("[IPS DEBUG] example logged actions (first 20):", ava[:20].tolist())
        print("[IPS DEBUG] rewards shape:", rva.shape)
        print("[IPS DEBUG] example rewards (first 5):", rva[:5].tolist())
    
    # Evaluate meta-selector performance
    meta_ips = estimate_ips(rva, ava, meta_actions)
    meta_snips = estimate_snips(rva, ava, meta_actions)
    
    print(f"[Info] Meta-selector IPS: {meta_ips:.4f}, SNIPS: {meta_snips:.4f}")
    
    # Log metrics for xgboost_simple
    if selector_type == "xgboost_simple":
        print(f"[XGB SIMPLE] val_ips_meta: {meta_ips:.4f}")
        print(f"[XGB SIMPLE] val_snips_meta: {meta_snips:.4f}")
    
    # Log match rate for static_multi_context and xgboost_simple
    if selector_type in ["static_multi_context", "xgboost_simple"]:
        meta_matches = (ava == meta_actions).sum()
        meta_match_rate = meta_matches / len(ava) if len(ava) > 0 else 0.0
        print(f"[IPS DEBUG] meta match rate: {meta_matches} / {len(ava)} = {meta_match_rate:.4f}")
    
    # Manual IPS consistency check (only for static_multi_context)
    if selector_type == "static_multi_context":
        slice_size = min(2000, len(rva))
        if slice_size > 0:
            slice_rva = rva[:slice_size]
            slice_ava = ava[:slice_size]
            slice_meta_actions = meta_actions[:slice_size]
            
            # Get rule and per_action actions for comparison
            slice_rule_actions = val_actions.get("rule", np.zeros(len(rva)))[:slice_size] if "rule" in val_actions else np.zeros(slice_size)
            slice_per_action_actions = val_actions.get("per_action", np.zeros(len(rva)))[:slice_size] if "per_action" in val_actions else np.zeros(slice_size)
            
            # Debug: Check matches
            rule_matches = (slice_ava == slice_rule_actions).sum()
            per_action_matches = (slice_ava == slice_per_action_actions).sum()
            meta_matches = (slice_ava == slice_meta_actions).sum()
            print(f"[IPS DEBUG] slice matches - rule: {rule_matches}/{slice_size}, per_action: {per_action_matches}/{slice_size}, meta: {meta_matches}/{slice_size}")
            
            # Debug: Check reward statistics for matches
            if rule_matches > 0:
                rule_match_rewards = slice_rva[slice_ava == slice_rule_actions]
                print(f"[IPS DEBUG] rule match rewards: mean={rule_match_rewards.mean():.4f}, std={rule_match_rewards.std():.4f}, count={len(rule_match_rewards)}")
            if meta_matches > 0:
                meta_match_rewards = slice_rva[slice_ava == slice_meta_actions]
                print(f"[IPS DEBUG] meta match rewards: mean={meta_match_rewards.mean():.4f}, std={meta_match_rewards.std():.4f}, count={len(meta_match_rewards)}")
            
            ips_rule_slice = estimate_ips(slice_rva, slice_ava, slice_rule_actions)
            ips_per_action_slice = estimate_ips(slice_rva, slice_ava, slice_per_action_actions)
            ips_meta_slice = estimate_ips(slice_rva, slice_ava, slice_meta_actions)
            
            print(f"[IPS DEBUG] slice IPS - rule: {ips_rule_slice:.4f}, per_action: {ips_per_action_slice:.4f}, meta: {ips_meta_slice:.4f}")
            
            # Also check full dataset matches
            full_rule_matches = (ava == val_actions.get("rule", np.zeros(len(ava)))).sum() if "rule" in val_actions else 0
            full_meta_matches = (ava == meta_actions).sum()
            print(f"[IPS DEBUG] full dataset matches - rule: {full_rule_matches}/{len(ava)}, meta: {full_meta_matches}/{len(ava)}")
            print(f"[IPS DEBUG] full dataset lengths - rva: {len(rva)}, ava: {len(ava)}, meta_actions: {len(meta_actions)}")
            
            if full_meta_matches > 0:
                full_meta_match_rewards = rva[ava == meta_actions]
                print(f"[IPS DEBUG] full meta match rewards: mean={full_meta_match_rewards.mean():.4f}, std={full_meta_match_rewards.std():.4f}, count={len(full_meta_match_rewards)}")
            
            # Debug: Check if there's an alignment issue by comparing first few actions
            print(f"[IPS DEBUG] First 10 ava: {ava[:10]}")
            print(f"[IPS DEBUG] First 10 meta_actions: {meta_actions[:10]}")
            print(f"[IPS DEBUG] First 10 matches: {(ava[:10] == meta_actions[:10]).tolist()}")
    
    # Build policy selection counts
    if selector_type in ["static_multi_context", "xgboost_simple"]:
        # Use the actual selected policies we computed above
        if 'meta_selected_policies' in locals():
            valid_policies = [p for p in meta_selected_policies if p is not None]
            if len(valid_policies) > 0:
                unique_policies, counts = np.unique(valid_policies, return_counts=True)
                policy_selections = {str(policy_name): int(count) for policy_name, count in zip(unique_policies, counts)}
            else:
                policy_selections = {}
        else:
            policy_selections = {}
        # Ensure all policy names are present (even if count is 0)
        for name in policy_names:
            if name not in policy_selections:
                policy_selections[name] = 0
    else:
        policy_selections = {
            name: np.sum([meta_selector.select(Xva[i], policy_rewards_va.iloc[i].values) == name 
                         for i in range(len(Xva))]) 
            for name in policy_names
        }
    
    metrics = {
        "val_ips_meta": meta_ips,
        "val_snips_meta": meta_snips,
        "meta_policy_selections": policy_selections
    }
    
    return meta_selector, metrics


@hydra.main(config_path=None, config_name=None)
def main(cfg: DictConfig):
    print("[Stage] Loading data...")
    raw = load_csv_with_limit(cfg.data.sample_path, n_rows=cfg.data.max_rows)
    print(f"[Info] Loaded rows: {len(raw)}")

    print("[Stage] Building features...")
    df = build_state_action_table(raw)
    print(f"[Info] Rows after lag features: {len(df)}")

    print("[Stage] Train/Val split...")
    train_df, val_df = patient_split(df, cfg.split.val_frac, cfg.seed)
    print(f"[Info] Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    feature_cols = BASE_FEATURES + [f"{c}_lag1" for c in BASE_FEATURES] + [f"{ACTION_COL}_lag1"]

    def to_arrays(d: pd.DataFrame):
        if len(d) == 0:
            return np.empty((0, len(feature_cols))), np.array([]), np.array([])
        X = d[feature_cols].values
        a = d[cfg.columns.action].values.astype(int)
        vl = d[cfg.columns.vl_out].values
        cd4 = d[cfg.columns.cd4_out].values
        r = make_reward(vl, cd4)
        return X, a, r

    Xtr, atr, rtr = to_arrays(train_df)
    Xva, ava, rva = to_arrays(val_df)

    print("[Stage] Training policies...")
    policies = {}
    
    if cfg.policies.rule:
        policies["rule"] = RuleBasedPolicy()
    
    if cfg.policies.per_action and len(Xtr) > 0:
        rf = RandomForestRegressor(n_estimators=cfg.model.n_estimators, random_state=cfg.seed)
        policies["per_action"] = PerActionSupervisedPolicy(rf, actions=sorted(np.unique(atr)), verbose=True).fit(Xtr, atr, rtr)
    
    if cfg.policies.dqn and len(Xtr) > 0:
        policies["dqn"] = DQNPolicy(actions=sorted(np.unique(atr)), verbose=True).fit(Xtr, atr, rtr)
    
    if cfg.policies.safety and len(Xtr) > 0:
        rf = RandomForestRegressor(n_estimators=cfg.model.n_estimators, random_state=cfg.seed)
        policies["safety"] = SafetyAwarePolicy(rf, actions=sorted(np.unique(atr)), safety_threshold=cfg.model.safety_threshold, verbose=True).fit(Xtr, atr, rtr)
    
    if cfg.policies.get("cf_knn", False) and len(Xtr) > 0:
        policies["cf_knn"] = CFKNNPolicy(actions=sorted(np.unique(atr)), n_neighbors=10, verbose=True).fit(Xtr, atr, rtr, df=train_df)

    print("[Stage] Generating validation actions...")
    val_actions, cf_meta_features_va = compute_policy_actions(policies, Xva, val_df)

    print("[Stage] Training reward model for DR...")
    if len(Xtr) > 0 and len(Xva) > 0:
        q_model = RandomForestRegressor(n_estimators=cfg.model.n_estimators, random_state=cfg.seed).fit(Xtr, rtr)
        q_hat_va = q_model.predict(Xva)
    else:
        q_hat_va = np.zeros(len(Xva)) if len(Xva) > 0 else np.array([])

    print("[Stage] Evaluating individual policies...")
    metrics = {}
    for name, targ_a in val_actions.items():
        if len(rva) == 0:
            metrics[f"val_ips_{name}"] = 0.0
            metrics[f"val_snips_{name}"] = 0.0
            metrics[f"val_dr_{name}"] = 0.0
            continue
        ips = estimate_ips(rva, ava, targ_a)
        snips = estimate_snips(rva, ava, targ_a)
        dr = estimate_dr(rva, ava, targ_a, q_hat=q_hat_va)
        metrics[f"val_ips_{name}"] = ips
        metrics[f"val_snips_{name}"] = snips
        metrics[f"val_dr_{name}"] = dr

    # Per-slice evaluation by history length quartiles
    if len(val_df) > 0:
        print("[Stage] Per-slice evaluation by history length...")
        # Calculate history length per patient (number of timesteps)
        history_lengths = val_df.groupby('PatientID').size()
        val_df_copy = val_df.copy()
        val_df_copy['history_length'] = val_df_copy['PatientID'].map(history_lengths)
        
        # Create quartiles
        quartiles = val_df_copy['history_length'].quantile([0.25, 0.5, 0.75])
        # Remove duplicates and ensure bins are unique
        unique_bins = [0]
        for q_val in [quartiles[0.25], quartiles[0.5], quartiles[0.75]]:
            if q_val not in unique_bins:
                unique_bins.append(q_val)
        unique_bins.append(float('inf'))
        
        if len(unique_bins) < 3:
            # Not enough unique quartiles, skip per-slice evaluation
            print("  [Warning] Not enough unique history lengths for quartile analysis. Skipping per-slice evaluation.")
        else:
            # Create labels based on number of bins
            n_bins = len(unique_bins) - 1
            labels = [f'Q{i+1}' for i in range(n_bins)]
            val_df_copy['history_quartile'] = pd.cut(
                val_df_copy['history_length'],
                bins=unique_bins,
                labels=labels,
                duplicates='drop'
            )
            
            # Evaluate policies per quartile (dynamic based on actual quartiles)
            quartile_labels = val_df_copy['history_quartile'].cat.categories if hasattr(val_df_copy['history_quartile'], 'cat') else val_df_copy['history_quartile'].unique()
            
            for quartile in quartile_labels:
                quartile_mask = val_df_copy['history_quartile'] == quartile
                if quartile_mask.sum() == 0:
                    continue
                
                quartile_indices = val_df_copy[quartile_mask].index
                quartile_Xva = Xva[quartile_indices] if len(Xva) > max(quartile_indices) else Xva
                quartile_ava = ava[quartile_indices] if len(ava) > max(quartile_indices) else ava
                quartile_rva = rva[quartile_indices] if len(rva) > max(quartile_indices) else rva
                
                if len(quartile_rva) == 0:
                    continue
                
                # Evaluate each policy in this quartile
                for name, targ_a in val_actions.items():
                    quartile_targ_a = targ_a[quartile_indices] if len(targ_a) > max(quartile_indices) else targ_a
                    if len(quartile_targ_a) == 0 or len(quartile_rva) == 0:
                        continue
                    
                    q_ips = estimate_ips(quartile_rva, quartile_ava, quartile_targ_a)
                    metrics[f"val_ips_{name}_{quartile.replace(' ', '_').replace('(', '').replace(')', '')}"] = q_ips
                    
                print(f"  {quartile}: {quartile_mask.sum()} samples")
    
    # Train and evaluate meta-selector
    meta_selector, meta_metrics = train_meta_selector(train_df, val_df, policies, feature_cols, cfg)
    metrics.update(meta_metrics)
    
    # Compute DR for meta-selector if q_hat is available
    # Note: meta_actions are computed in train_meta_selector, but we need them here for DR
    # For simplicity, we'll store them in the selector or recompute if needed
    selector_type = cfg.meta.get("selector_type", "rf")
    if len(rva) > 0 and len(ava) > 0 and hasattr(meta_selector, '_meta_actions_aligned'):
        # Use stored actions if available
        meta_actions_for_dr = meta_selector._meta_actions_aligned
        if len(meta_actions_for_dr) == len(rva):
            meta_dr = estimate_dr(rva, ava, meta_actions_for_dr, q_hat=q_hat_va)
            metrics["val_dr_meta"] = meta_dr
            # Log DR for xgboost_simple
            if selector_type == "xgboost_simple":
                print(f"[XGB SIMPLE] val_dr_meta: {meta_dr:.4f}")

    # Save meta-selector model
    import pickle
    from pathlib import Path
    
    model_dir = Path("artifacts")
    model_dir.mkdir(exist_ok=True)
    
    selector_type = cfg.meta.get("selector_type", "rf")
    model_path = model_dir / f"meta_selector_{selector_type}.pkl"
    
    print(f"[Info] Saving meta-selector to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(meta_selector, f)
    print(f"[Info] Meta-selector saved successfully")

    print("Config:\n", OmegaConf.to_yaml(cfg))
    for k, v in metrics.items():
        if k.startswith("meta_policy_selections"):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")

    print("[Stage] Logging to MLflow...")
    mlflow.set_tracking_uri(f"file://{os.path.abspath(cfg.logging.mlflow_dir)}")
    mlflow.set_experiment(cfg.logging.experiment)
    with mlflow.start_run(run_name=cfg.logging.run_name):
        params_to_log = {
            "seed": cfg.seed,
            "val_frac": cfg.split.val_frac,
            "n_estimators": cfg.model.n_estimators,
            "pol_rule": cfg.policies.rule,
            "pol_per_action": cfg.policies.per_action,
            "pol_dqn": cfg.policies.dqn,
            "pol_safety": cfg.policies.safety,
            "pol_cf_knn": cfg.policies.get("cf_knn", False),
            "max_rows": cfg.data.max_rows,
            "meta_use_patient_features": cfg.meta.use_patient_features,
            "meta_model_type": cfg.meta.model_type,
            "meta_selector_type": cfg.meta.get("selector_type", "rf"),
            "safety_threshold": cfg.model.safety_threshold,
        }
        
        # Add transformer_gb-specific params if using transformer_gb
        if cfg.meta.get("selector_type") == "transformer_gb":
            params_to_log.update({
                "transformer_hidden_size": cfg.meta.get("transformer_hidden_size", 128),
                "transformer_num_heads": cfg.meta.get("transformer_num_heads", 4),
                "transformer_num_layers": cfg.meta.get("transformer_num_layers", 2),
                "transformer_dropout": cfg.meta.get("transformer_dropout", 0.25),
                "transformer_lr": cfg.meta.get("transformer_lr", 0.001),
                "transformer_batch_size": cfg.meta.get("transformer_batch_size", 64),
                "transformer_epochs": cfg.meta.get("transformer_epochs", 50),
                "transformer_patience": cfg.meta.get("transformer_patience", 8),
                "transformer_grad_clip": cfg.meta.get("transformer_grad_clip", 1.0),
                "gb_backend": cfg.meta.get("gb_backend", "xgboost"),
            })
        
        # Add static multi-context-specific params if using static_multi_context
        if cfg.meta.get("selector_type") == "static_multi_context":
            params_to_log.update({
                "static_d_model": cfg.meta.get("static_d_model", 128),
                "static_nheads": cfg.meta.get("static_nheads", 4),
                "static_nlayers": cfg.meta.get("static_nlayers", 2),
                "static_dim_feedforward": cfg.meta.get("static_dim_feedforward", 256),
                "static_dropout": cfg.meta.get("static_dropout", 0.15),
                "static_lr": cfg.meta.get("static_lr", 0.001),
                "static_batch_size": cfg.meta.get("static_batch_size", 64),
                "static_epochs": cfg.meta.get("static_epochs", 25),
                "static_patience": cfg.meta.get("static_patience", 5),
                "static_min_delta": cfg.meta.get("static_min_delta", 1e-4),
                "static_use_class_weights": cfg.meta.get("static_use_class_weights", True),
                "static_reward_normalization": cfg.meta.get("static_reward_normalization", "zscore"),
            })
        
        mlflow.log_params(params_to_log)
        # Log metrics (excluding the dict)
        for k, v in metrics.items():
            if not k.startswith("meta_policy_selections"):
                mlflow.log_metric(k, v)
    print("[Done]")


if __name__ == "__main__":
    default_cfg = {
        "seed": 0,
        "data": {"sample_path": "dataset/HealthGymV2_CbdrhDatathon_ART4HIV.csv", "max_rows": None},
        "columns": {"action": "Drug (M)", "vl_out": "VL (M)", "cd4_out": "CD4 (M)"},
        "split": {"val_frac": 0.3},
        "model": {"n_estimators": 50, "safety_threshold": 0.3},
        "policies": {"rule": True, "per_action": True, "dqn": True, "safety": True, "cf_knn": False},
        "meta": {
            "use_patient_features": True,
            "model_type": "rf",
            "selector_type": "static_multi_context",  # "transformer_gb", "static_multi_context", or "xgboost_simple"
            # Transformer-GB specific hyperparameters (only used if selector_type == "transformer_gb")
            "transformer_hidden_size": 128,
            "transformer_num_heads": 4,
            "transformer_num_layers": 2,
            "transformer_dropout": 0.25,
            "transformer_lr": 0.001,
            "transformer_batch_size": 64,
            "transformer_epochs": 50,
            "transformer_patience": 8,
            "transformer_min_delta": 1e-4,
            "transformer_grad_clip": 1.0,
            "gb_backend": "xgboost",  # For transformer_gb: "xgboost" or "lightgbm"
            # XGBoost Simple specific hyperparameters (only used if selector_type == "xgboost_simple")
            "xgb_simple_max_depth": 4,
            "xgb_simple_n_estimators": 200,
            "xgb_simple_learning_rate": 0.05,
            "xgb_simple_subsample": 0.8,
            "xgb_simple_colsample_bytree": 0.8,
            "xgb_simple_reg_lambda": 1.0,
            "xgb_simple_reg_alpha": 0.0,
            "xgb_simple_use_class_weights": True,
            # Static multi-context specific hyperparameters (only used if selector_type == "static_multi_context")
            "static_d_model": 128,
            "static_nheads": 4,
            "static_nlayers": 2,
            "static_dim_feedforward": 256,
            "static_dropout": 0.15,
            "static_lr": 0.001,
            "static_batch_size": 64,
            "static_epochs": 25,
            "static_patience": 5,
            "static_min_delta": 1e-4,
            "static_use_class_weights": True,
            "static_reward_normalization": "zscore",  # "none", "zscore", "relative"
            "device": "cpu",  # "cpu" or "cuda" if GPU available
        },
        "logging": {"mlflow_dir": "mlruns", "experiment": "meta-recommender", "run_name": "sample-run"},
    }
    main(OmegaConf.create(default_cfg))
