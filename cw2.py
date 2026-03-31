"""
CW2: Neuro-Symbolic AI System
Student Name: [Your Name]
Student ID: [Your ID]

This module implements a neuro-symbolic AI system that combines:
- Computer Vision (CIFAR-100 object recognition)
- Natural Language Processing (Skip-gram word embeddings)
- Symbolic Planning (PDDL planning)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Dict, Tuple, Optional
from pathlib import Path
import warnings

# lab imports


# ============================================================================
# SECTION 1: CIFAR-100 SEMANTIC EXPANSION
# ============================================================================

# DO NOT CHANGE THIS FUNCTION's signature
def build_my_embeddings(checkpoint_path: str = "best_skipgram_523words.pth") -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load and return your trained Skip-gram embeddings.
    
    This function serves as the entry point for loading your final embedding model
    that contains all Visual Genome words AND all 100 CIFAR-100 classes.
    
    Args:
        checkpoint_path: Path to your saved model checkpoint
        
    Returns:
        vocab: Dictionary mapping words to indices {word: index}
        embeddings: Numpy array of shape (vocab_size, embedding_dim)
        
    Example:
        >>> vocab, embeddings = build_my_embeddings()
        >>> print(f"Vocabulary size: {len(vocab)}")
        >>> print(f"Embedding dimension: {embeddings.shape[1]}")
        >>> print(f"'airplane' index: {vocab.get('airplane', 'NOT FOUND')}")
    """
    # TODO: Implement this function
    # 1. Load your checkpoint file
    # 2. Extract the vocabulary dictionary
    # 3. Extract the embedding matrix
    # 4. Ensure vocabulary contains all required words (Visual Genome + CIFAR-100)    
    return None


# ============================================================================
# SECTION 2: NEURO-SYMBOLIC AI - MULTI-MODAL PLANNING
# ============================================================================

# DO NOT CHANGE THIS FUNCTION's signature
def plan_generator(input_data: Union[torch.Tensor, str],    # ASSUME default CIFAR-100 image dimensions
                  initial_state: List[str],                 # Consistent with Lab9 syntax
                  goal_state: List[str],                    # Consistent with Lab9 syntax
                  domain_file: str = "domain.pddl",
                  skipgram_path: str = "best_skipgram_523words.pth",
                  projection_path: str = "best_cifar100_projection.pth") -> Optional[List[str]]:
    """
    !!!WARNING!!!: Treat this as pseudocode. You may need to modify the logic. 
    
    Main entry point for the neuro-symbolic planning system.
    
    This function implements the complete pipeline from perception to planning.
    
    Args:
        input_data: Either an image tensor OR object name string
        initial_state: List of predicates describing initial state                      
        goal_state: List of predicates describing goal state                   
        domain_file: Path to the PDDL domain file
        skipgram_path: Path to Skip-gram embeddings checkpoint
        projection_path: Path to CIFAR-100 projection model checkpoint
        
    Returns:
        A list of action strings representing the plan, 
            OR None if:
                - The object cannot be identified
                - No valid plan exists
                - ...
        
    Example:
        >>> image = # CIFAR-100 image
        >>> initial = ["on table"]
        >>> goal = ["in basket"]
        >>> plan = plan_generator(image, initial, goal, "domain.pddl")        
    """
    
    # TREAT THIS AS SUGGESTED PSEUDOCODE. YOU MAY USE OTHER PARADIGMS
                    
    # Step 0: Initialize the planner
    
    # Step 1: Identify the object
    
    # Step 2: Parse PDDL domain
    
    # Step 3: Create PDDL problem
    
    # Step 4: Generate plan
        
    return None
