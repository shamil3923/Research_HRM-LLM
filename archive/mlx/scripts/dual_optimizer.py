"""
Dual optimizer support for separate embedding learning rates
Matches the original PyTorch HRM implementation with two optimizers
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_adam_atan2_exact import AdamATan2Exact
from typing import Dict, Any, List, Tuple


class DualAdamATan2:
    """
    Dual optimizer setup matching original PyTorch HRM:
    - AdamATan2 for main parameters (base learning rate)
    - AdamATan2 for embedding parameters (separate learning rate)
    
    This mimics the original's dual optimizer approach:
    - AdamATan2 for model.parameters()
    - CastedSparseEmbeddingSignSGD_Distributed for model.puzzle_emb.buffers()
    """
    
    def __init__(
        self,
        base_lr: float = 1e-4,
        embedding_lr: float = 1e-2,  # Original uses 1e-2, not 7e-5
        weight_decay: float = 0.1,
        embedding_weight_decay: float = 0.1,
        betas: tuple = (0.9, 0.95),
        a: float = 1.27,
        b: float = 1.0,
    ):
        self.base_lr = base_lr
        self.embedding_lr = embedding_lr
        self.weight_decay = weight_decay
        self.embedding_weight_decay = embedding_weight_decay
        
        # Create two separate AdamATan2 optimizers
        self.main_optimizer = AdamATan2Exact(
            learning_rate=base_lr,
            weight_decay=weight_decay,
            betas=betas,
            a=a,
            b=b,
        )
        
        self.embedding_optimizer = AdamATan2Exact(
            learning_rate=embedding_lr,
            weight_decay=embedding_weight_decay,
            betas=betas,
            a=a,
            b=b,
        )
        
        self.param_groups = None
        self.learning_rate = base_lr  # For compatibility with LR scheduler
        
        print(f"🔧 Dual optimizer setup:")
        print(f"   Main parameters: lr={base_lr}, wd={weight_decay}")
        print(f"   Embedding parameters: lr={embedding_lr}, wd={embedding_weight_decay}")
    
    def init(self, parameters: Dict[str, Any]):
        """Initialize both optimizers with separated parameter groups"""
        # Separate parameters into main and embedding groups
        main_params, embedding_params = self._separate_parameters(parameters)
        
        # Initialize both optimizers
        if main_params:
            self.main_optimizer.init(main_params)
        if embedding_params:
            self.embedding_optimizer.init(embedding_params)
        
        # Store parameter groups for later use
        self.param_groups = {
            'main': main_params,
            'embedding': embedding_params
        }
        
        print(f"📊 Parameter separation:")
        print(f"   Main parameters: {len(main_params)} groups")
        print(f"   Embedding parameters: {len(embedding_params)} groups")
    
    def _separate_parameters(self, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Separate parameters into main and embedding groups"""
        main_params = {}
        embedding_params = {}
        
        for name, param in parameters.items():
            # Check if this is an embedding parameter
            if self._is_embedding_param(name):
                embedding_params[name] = param
            else:
                main_params[name] = param
        
        return main_params, embedding_params
    
    def _is_embedding_param(self, param_name: str) -> bool:
        """Determine if a parameter belongs to embedding layers"""
        embedding_keywords = [
            'embed_tokens',  # Main token embeddings
            'embed_pos',     # Position embeddings
            'puzzle_emb',    # Puzzle embeddings (if enabled)
        ]
        
        return any(keyword in param_name for keyword in embedding_keywords)
    
    def update(self, model, gradients: Dict[str, Any]):
        """Update parameters using both optimizers"""
        # Separate gradients into main and embedding groups
        main_grads, embedding_grads = self._separate_gradients(gradients)
        
        # Update main parameters
        if main_grads and self.param_groups['main']:
            # Create temporary model with main parameters only
            main_model_dict = {k: v for k, v in model.trainable_parameters().items() 
                              if k in self.param_groups['main']}
            self.main_optimizer.update(model, main_grads)
        
        # Update embedding parameters  
        if embedding_grads and self.param_groups['embedding']:
            # Create temporary model with embedding parameters only
            embedding_model_dict = {k: v for k, v in model.trainable_parameters().items() 
                                   if k in self.param_groups['embedding']}
            self.embedding_optimizer.update(model, embedding_grads)
    
    def _separate_gradients(self, gradients: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Separate gradients into main and embedding groups"""
        main_grads = {}
        embedding_grads = {}
        
        def separate_nested_dict(grads_dict, main_dict, emb_dict, path=""):
            for key, value in grads_dict.items():
                current_path = f"{path}.{key}" if path else key
                
                if isinstance(value, dict):
                    # Recursively handle nested dictionaries
                    if key not in main_dict:
                        main_dict[key] = {}
                    if key not in emb_dict:
                        emb_dict[key] = {}
                    separate_nested_dict(value, main_dict[key], emb_dict[key], current_path)
                else:
                    # This is a leaf gradient
                    if self._is_embedding_param(current_path):
                        emb_dict[key] = value
                    else:
                        main_dict[key] = value
        
        separate_nested_dict(gradients, main_grads, embedding_grads)
        return main_grads, embedding_grads
    
    def update_learning_rate(self, new_lr: float):
        """Update learning rate for main optimizer (for LR scheduler compatibility)"""
        self.learning_rate = new_lr
        self.main_optimizer.learning_rate = new_lr
        # Keep embedding LR at its fixed ratio or fixed value
        # For now, keep embedding LR constant
    
    @property
    def state(self):
        """Combined state from both optimizers"""
        return {
            'main': getattr(self.main_optimizer, 'state', {}),
            'embedding': getattr(self.embedding_optimizer, 'state', {})
        }