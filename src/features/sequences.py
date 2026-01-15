"""
Sequence building utilities for temporal patient trajectories.

This module provides functions to build time-ordered sequences of patient states
for use in sequence-based meta-selectors.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from src.features.state import ID_COL, TIME_COL


def build_patient_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build temporal sequences from patient trajectories.
    
    For each patient-timestep, extracts the last L timesteps of state features.
    Sequences are padded with zeros if a patient has fewer than L timesteps.
    
    Args:
        df: DataFrame with patient data, must have PatientID and Timestep columns
        feature_cols: List of feature column names to include in sequences
        sequence_length: Number of timesteps L to include in each sequence (default: 5)
    
    Returns:
        seq_x: Array of shape (n_samples, L, seq_input_dim) containing sequences
        indices: Array of shape (n_samples,) mapping to original dataframe row indices
    """
    # Ensure dataframe is sorted by PatientID and Timestep
    df = df.sort_values([ID_COL, TIME_COL]).copy().reset_index(drop=True)
    
    # Get unique patients
    patients = df[ID_COL].unique()
    
    sequences = []
    indices = []
    
    for patient_id in patients:
        patient_data = df[df[ID_COL] == patient_id].copy()
        patient_data = patient_data.sort_values(TIME_COL).reset_index(drop=True)
        
        # Extract feature values for this patient
        patient_features = patient_data[feature_cols].values  # (n_timesteps, n_features)
        
        # For each timestep in this patient's trajectory
        for i in range(len(patient_data)):
            # Get the last L timesteps up to and including current timestep
            start_idx = max(0, i - sequence_length + 1)
            end_idx = i + 1
            
            # Extract sequence
            seq = patient_features[start_idx:end_idx]  # (actual_length, n_features)
            
            # Pad with zeros if sequence is shorter than L
            if len(seq) < sequence_length:
                padding = np.zeros((sequence_length - len(seq), seq.shape[1]))
                seq = np.vstack([padding, seq])
            
            # Ensure exactly L timesteps (take last L if somehow longer)
            if len(seq) > sequence_length:
                seq = seq[-sequence_length:]
            
            sequences.append(seq)
            # Store the original dataframe index for this row
            original_idx = patient_data.index[i]
            indices.append(original_idx)
    
    # Convert to numpy arrays
    seq_x = np.array(sequences)  # (n_samples, L, seq_input_dim)
    indices = np.array(indices)  # (n_samples,)
    
    return seq_x, indices

