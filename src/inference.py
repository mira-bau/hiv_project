"""
Canonical inference functions for medical case UI.

Provides standardized functions to load models and make predictions
using the final static multi-context meta-selector.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import pickle


def load_medical_models(artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the final meta-selector and all base policies from artifacts.
    
    Args:
        artifacts_dir: Path to artifacts directory. If None, uses default.
    
    Returns:
        Dict with keys:
            - 'meta_selector': StaticMultiContextMetaSelector instance
            - 'policies': Dict mapping policy names to policy instances
    
    Raises:
        FileNotFoundError: If required artifacts are missing (with exact paths)
    """
    if artifacts_dir is None:
        # Default: assume called from medical_case/app/ directory
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
    else:
        artifacts_dir = Path(artifacts_dir)
    
    # Load meta-selector
    meta_selector_path = artifacts_dir / "meta_selector_static_multi_context.pkl"
    if not meta_selector_path.exists():
        raise FileNotFoundError(
            f"Meta-selector not found at: {meta_selector_path.absolute()}\n"
            f"Please train the model first using: python run_static_multi_context_full.py --rerun"
        )
    
    try:
        with open(meta_selector_path, 'rb') as f:
            meta_selector = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load meta-selector from {meta_selector_path}: {e}")
    
    # Load base policies
    policies_path = artifacts_dir / "policies.pkl"
    if not policies_path.exists():
        raise FileNotFoundError(
            f"Base policies not found at: {policies_path.absolute()}\n"
            f"Please train the model first using: python run_static_multi_context_full.py --rerun"
        )
    
    try:
        with open(policies_path, 'rb') as f:
            policies = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load policies from {policies_path}: {e}")
    
    # Verify policies dict structure
    if not isinstance(policies, dict):
        raise ValueError(f"Expected policies to be a dict, got {type(policies)}")
    
    expected_policies = ['rule', 'per_action', 'dqn', 'safety', 'cf_knn']
    missing = [p for p in expected_policies if p not in policies]
    if missing:
        raise ValueError(f"Missing required policies: {missing}. Found: {list(policies.keys())}")
    
    return {
        'meta_selector': meta_selector,
        'policies': policies
    }


def build_features_for_inference(
    patient_state_vec: np.ndarray,
    policies: Dict[str, Any],
    df_row: Optional[pd.DataFrame],
    feature_cols: List[str],
    mean_reward: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, Any]]]:
    """
    Build features needed for static multi-context meta-selector inference.
    
    Args:
        patient_state_vec: State feature vector (state_dim,)
        policies: Dict of policy instances
        df_row: DataFrame row for CF-kNN (can be None)
        feature_cols: List of feature column names
        mean_reward: Mean reward to use as fallback (default 0.0)
    
    Returns:
        Tuple of:
            - state_x: (1, state_dim) state features
            - policy_feats: (1, num_policies, 1) policy reward predictions
            - neighbor_feats: (1, 3) CF neighbor features
            - base_policy_outputs: Dict with action, predicted_reward, confidence for each policy
    """
    # Ensure state features are correct shape
    if patient_state_vec.ndim == 1:
        state_x = patient_state_vec.reshape(1, -1)
    else:
        state_x = patient_state_vec
    
    state_dim = state_x.shape[1]
    
    # Policy names in order
    policy_names = ['rule', 'per_action', 'dqn', 'safety', 'cf_knn']
    n_policies = len(policy_names)
    
    # Build policy features and outputs
    policy_feats = np.zeros((1, n_policies, 1))
    base_policy_outputs = {}
    
    # Get actions and predicted rewards from each policy
    for idx, policy_name in enumerate(policy_names):
        policy = policies.get(policy_name)
        if policy is None:
            action = 0
            predicted_reward = mean_reward
            confidence = 0.0
        else:
            # Get action recommendation
            if policy_name == 'rule':
                action = policy.recommend({})
                # Rule-based policy doesn't predict reward, use mean
                predicted_reward = mean_reward
                confidence = 1.0  # Rule-based is deterministic
            elif policy_name == 'cf_knn':
                action = policy.recommend(patient_state_vec, df_row)
                # Get meta-features for CF-kNN
                meta = policy.get_meta_features()
                predicted_reward = meta.get('cf_predicted_reward', mean_reward)
                confidence = meta.get('cf_confidence', 0.5)
            else:
                # per_action, dqn, safety
                action = policy.recommend(patient_state_vec)
                # Try to get predicted reward from policy's internal model
                predicted_reward = mean_reward
                confidence = 0.5
                
                # For per_action, try to get prediction from model for the recommended action
                if policy_name == 'per_action' and hasattr(policy, 'models'):
                    try:
                        # Get prediction for the recommended action
                        if action in policy.models:
                            model = policy.models[action]
                            if hasattr(model, 'predict'):
                                predicted_reward = float(model.predict(patient_state_vec.reshape(1, -1))[0])
                        else:
                            # If action not in models, try to get max prediction across all actions
                            max_reward = -1e9
                            for a, model in policy.models.items():
                                if hasattr(model, 'predict'):
                                    reward = float(model.predict(patient_state_vec.reshape(1, -1))[0])
                                    max_reward = max(max_reward, reward)
                            if max_reward > -1e9:
                                predicted_reward = max_reward
                    except Exception:
                        pass
                
                # For dqn, try to get Q-value for the recommended action
                elif policy_name == 'dqn' and hasattr(policy, 'q_models'):
                    try:
                        # Get Q-value for the recommended action
                        if action in policy.q_models:
                            q_model = policy.q_models[action]
                            if hasattr(q_model, 'predict'):
                                predicted_reward = float(q_model.predict(patient_state_vec.reshape(1, -1))[0])
                        else:
                            # If action not in q_models, try to get max Q-value
                            max_q = -1e9
                            for a, q_model in policy.q_models.items():
                                if hasattr(q_model, 'predict'):
                                    q_val = float(q_model.predict(patient_state_vec.reshape(1, -1))[0])
                                    max_q = max(max_q, q_val)
                            if max_q > -1e9:
                                predicted_reward = max_q
                    except Exception:
                        pass
                
                # For safety, try to get reward prediction for the recommended action
                elif policy_name == 'safety' and hasattr(policy, 'reward_models'):
                    try:
                        # Get reward prediction for the recommended action
                        if action in policy.reward_models:
                            reward_model = policy.reward_models[action]
                            if hasattr(reward_model, 'predict'):
                                predicted_reward = float(reward_model.predict(patient_state_vec.reshape(1, -1))[0])
                        else:
                            # If action not in reward_models, try to get max reward
                            max_reward = -1e9
                            for a, reward_model in policy.reward_models.items():
                                if hasattr(reward_model, 'predict'):
                                    reward = float(reward_model.predict(patient_state_vec.reshape(1, -1))[0])
                                    max_reward = max(max_reward, reward)
                            if max_reward > -1e9:
                                predicted_reward = max_reward
                    except Exception:
                        pass
            
            # Store policy output
            base_policy_outputs[policy_name] = {
                'action': int(action),
                'predicted_reward': float(predicted_reward),
                'confidence': float(confidence)
            }
        
        # Store in policy_feats array
        policy_feats[0, idx, 0] = predicted_reward
    
    # Apply per-sample z-score normalization to policy rewards (matching training)
    policy_rewards = policy_feats[0, :, 0]
    mean_rewards = np.mean(policy_rewards)
    std_rewards = np.std(policy_rewards)
    if std_rewards < 1e-6:
        std_rewards = 1e-6
    normalized_rewards = (policy_rewards - mean_rewards) / std_rewards
    policy_feats[0, :, 0] = normalized_rewards
    
    # Build neighbor features from CF-kNN
    neighbor_feats = np.zeros((1, 3))
    cf_policy = policies.get('cf_knn')
    if cf_policy is not None and hasattr(cf_policy, 'get_meta_features'):
        try:
            meta = cf_policy.get_meta_features()
            neighbor_feats[0, 0] = meta.get('cf_predicted_reward', 0.0)
            neighbor_feats[0, 1] = float(meta.get('cf_neighbor_density', 0))
            neighbor_feats[0, 2] = meta.get('cf_confidence', 0.5)
        except:
            # Default values if CF features not available
            neighbor_feats[0, 0] = 0.0
            neighbor_feats[0, 1] = 0.0
            neighbor_feats[0, 2] = 0.5
    else:
        # Default values if CF-kNN not available
        neighbor_feats[0, 0] = 0.0
        neighbor_feats[0, 1] = 0.0
        neighbor_feats[0, 2] = 0.5
    
    return state_x, policy_feats, neighbor_feats, base_policy_outputs


def predict_for_patient(
    patient_id: Any,
    visit_idx: Optional[int],
    models: Dict[str, Any],
    raw_data: pd.DataFrame,
    processed_data: pd.DataFrame,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Make prediction for a patient using the final meta-selector.
    
    Args:
        patient_id: Patient ID
        visit_idx: Visit/timestep index (None for latest)
        models: Dict from load_medical_models() with 'meta_selector' and 'policies'
        raw_data: Raw patient data (for CF-kNN)
        processed_data: Processed data with features
        feature_cols: List of feature column names
    
    Returns:
        Structured dict with:
            - patient_summary: Basic patient info
            - labs: VL, CD4 values
            - base_policy_outputs: Actions and predictions for each policy
            - meta_output: Selected policy, action, probabilities, confidence
    """
    meta_selector = models['meta_selector']
    policies = models['policies']
    
    # Get patient data
    patient_processed = processed_data[processed_data['PatientID'] == patient_id].copy()
    if len(patient_processed) == 0:
        raise ValueError(f"Patient {patient_id} not found in processed data")
    
    # Get specific visit or latest
    if visit_idx is not None:
        patient_row = patient_processed[patient_processed['Timestep'] == visit_idx]
        if len(patient_row) == 0:
            raise ValueError(f"Visit {visit_idx} not found for patient {patient_id}")
        patient_row = patient_row.iloc[0]
    else:
        patient_row = patient_processed.iloc[-1]
    
    # Get corresponding raw data row for CF-kNN
    patient_raw = raw_data[raw_data['PatientID'] == patient_id].copy()
    if len(patient_raw) > 0:
        if visit_idx is not None:
            raw_row = patient_raw[patient_raw['Timestep'] == visit_idx]
            if len(raw_row) > 0:
                raw_row = raw_row.iloc[[0]]
            else:
                raw_row = patient_raw.iloc[[-1]]
        else:
            raw_row = patient_raw.iloc[[-1]]
    else:
        raw_row = None
    
    # Extract state features
    available_cols = [col for col in feature_cols if col in patient_row.index]
    missing_cols = [col for col in feature_cols if col not in patient_row.index]
    
    if missing_cols:
        # Fill missing columns with 0
        for col in missing_cols:
            patient_row[col] = 0.0
    
    patient_state_vec = patient_row[feature_cols].values
    
    # Build features for inference
    state_x, policy_feats, neighbor_feats, base_policy_outputs = build_features_for_inference(
        patient_state_vec, policies, raw_row, feature_cols
    )
    
    # Get meta-selector prediction
    selected_policy_name = meta_selector.select(state_x, policy_feats, neighbor_feats)
    
    # Get policy probabilities if available
    try:
        policy_probs = meta_selector.get_policy_probabilities(state_x, policy_feats, neighbor_feats)
        confidence = policy_probs.get(selected_policy_name, 0.0) * 100.0
    except:
        policy_probs = {name: 0.0 for name in meta_selector.policy_names}
        policy_probs[selected_policy_name] = 1.0
        confidence = 100.0
    
    # Get selected policy's action
    selected_policy = policies.get(selected_policy_name)
    if selected_policy is None:
        selected_action = 0
    else:
        if selected_policy_name == 'rule':
            selected_action = selected_policy.recommend({})
        elif selected_policy_name == 'cf_knn':
            selected_action = selected_policy.recommend(patient_state_vec, raw_row)
        else:
            selected_action = selected_policy.recommend(patient_state_vec)
    
    # Build result dict
    result = {
        'patient_summary': {
            'patient_id': patient_id,
            'visit_idx': visit_idx if visit_idx is not None else int(patient_row['Timestep']),
            'gender': patient_row.get('Gender', 'Unknown'),
            'ethnicity': patient_row.get('Ethnic', 'Unknown'),
        },
        'labs': {
            'vl': float(patient_row.get('VL', 0.0)),
            'cd4': float(patient_row.get('CD4', 0.0)),
        },
        'base_policy_outputs': base_policy_outputs,
        'meta_output': {
            'selected_policy_name': selected_policy_name,
            'selected_action': int(selected_action),
            'policy_probs': policy_probs,
            'confidence': float(confidence)
        }
    }
    
    return result

