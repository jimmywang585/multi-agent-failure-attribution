"""
Step 2: Discriminator class and initialization

Implements trainable discriminators that map log sequences to step probability distributions.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional, Tuple


class StepEncoder(nn.Module):
    """
    Encoder that converts a list of steps into embeddings using HuggingFace transformers.
    
    Takes a list of step objects (dicts or strings) and returns
    a tensor of shape [num_steps, d_model].
    """
    
    def __init__(self, encoder_type: str = "distilbert-base-uncased", max_length: int = 512, d_model: Optional[int] = None):
        """
        Args:
            encoder_type: HuggingFace model identifier (e.g., "distilbert-base-uncased")
            max_length: Maximum sequence length for tokenizer
            d_model: Output embedding dimension (if None, uses model's hidden size)
        """
        super().__init__()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_type)
        self.model = AutoModel.from_pretrained(encoder_type)
        
        # Set model to eval mode by default (will be set to train mode when needed)
        self.model.eval()
        
        # Get model's hidden size
        model_hidden_size = self.model.config.hidden_size
        
        # Set d_model (use model's hidden size if not specified)
        self.d_model = d_model if d_model is not None else model_hidden_size
        self.max_length = max_length
        
        # Projection layer if d_model differs from model hidden size
        if self.d_model != model_hidden_size:
            self.projection = nn.Linear(model_hidden_size, self.d_model)
        else:
            self.projection = nn.Identity()
    
    def _step_to_text(self, step: Any) -> str:
        """
        Convert a step object to text string.
        
        Args:
            step: A step object (dict with keys like 'content', 'role', etc., or a string)
            
        Returns:
            Text string representation
        """
        if isinstance(step, dict):
            # Extract text content from common keys
            text = step.get('content', step.get('text', step.get('message', step.get('role', str(step)))))
            if not isinstance(text, str):
                text = str(step)
        elif isinstance(step, str):
            text = step
        else:
            text = str(step)
        
        return text
    
    def forward(self, steps: List[Any]) -> torch.Tensor:
        """
        Encode a list of steps into embeddings.
        
        Args:
            steps: List of step objects (dicts or strings)
            
        Returns:
            Tensor of shape [num_steps, d_model]
        """
        if not steps:
            # Return empty tensor with correct shape
            device = next(self.model.parameters()).device
            return torch.zeros(0, self.d_model, device=device)
        
        device = next(self.model.parameters()).device
        embeddings = []
        
        # Set model to training mode if discriminator is training
        was_training = self.model.training
        if self.training:
            self.model.train()
        else:
            self.model.eval()
        
        try:
            for step in steps:
                # Convert step to text
                text = self._step_to_text(step)
                
                # Tokenize
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move to model device
                encoded = {k: v.to(device) for k, v in encoded.items()}
                
                # Get embeddings from model
                # Model parameters are trainable, so gradients flow through
                outputs = self.model(**encoded)
                
                # Extract embeddings
                # Use [CLS] token embedding or mean pool over sequence
                if hasattr(outputs, 'last_hidden_state'):
                    # Mean pool over sequence tokens (excluding padding)
                    # For simplicity, we'll use mean over all tokens
                    step_embed = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # [hidden_size]
                elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    # Use pooler output if available
                    step_embed = outputs.pooler_output.squeeze(0)  # [hidden_size]
                else:
                    # Fallback: use first token
                    step_embed = outputs[0][:, 0, :].squeeze(0)  # [hidden_size]
                
                # Project to d_model if needed
                step_embed = self.projection(step_embed)  # [d_model]
                
                embeddings.append(step_embed)
        finally:
            # Restore model training state
            if was_training:
                self.model.train()
            else:
                self.model.eval()
        
        # Stack into [num_steps, d_model]
        step_embeddings = torch.stack(embeddings, dim=0)
        
        return step_embeddings


class StepScorer(nn.Module):
    """
    Scorer that maps step embeddings to logits (raw scores).
    
    Takes [num_steps, d_model] and returns [num_steps] logits.
    """
    
    def __init__(self, d_model: int = 128):
        """
        Args:
            d_model: Dimension of input embeddings
        """
        super().__init__()
        # Single linear layer to score each step
        self.scorer = nn.Linear(d_model, 1)
    
    def forward(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Score step embeddings to produce logits.
        
        Args:
            step_embeddings: Tensor of shape [num_steps, d_model]
            
        Returns:
            Tensor of shape [num_steps] with logits
        """
        # Score each step: [num_steps, d_model] -> [num_steps, 1]
        logits = self.scorer(step_embeddings)  # [num_steps, 1]
        
        # Squeeze to [num_steps]
        logits = logits.squeeze(-1)
        
        return logits


class Discriminator(nn.Module):
    """
    Trainable discriminator that maps a log sequence to a probability distribution over steps.
    
    Architecture:
    1. Encoder: steps → embeddings [num_steps, d_model]
    2. Scorer: embeddings → logits [num_steps]
    3. Softmax: logits → probabilities [num_steps]
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize discriminator with encoder and scorer.
        
        Args:
            model_config: Dictionary with configuration parameters:
                - encoder_type: HuggingFace model ID (e.g., "distilbert-base-uncased")
                - max_length: Maximum sequence length (default: 512)
                - d_model: Embedding dimension (if None, uses model's hidden size)
        """
        super().__init__()
        
        # Default configuration
        if model_config is None:
            model_config = {}
        
        encoder_type = model_config.get('encoder_type', 'distilbert-base-uncased')
        max_length = model_config.get('max_length', 512)
        d_model = model_config.get('d_model', None)  # None = use model's hidden size
        
        # Build encoder: steps → embeddings (uses HuggingFace transformers)
        self.encoder = StepEncoder(encoder_type=encoder_type, max_length=max_length, d_model=d_model)
        
        # Get actual d_model from encoder (may differ from config if using model's hidden size)
        actual_d_model = self.encoder.d_model
        
        # Build scorer: embeddings → logits
        self.scorer = StepScorer(d_model=actual_d_model)
        
        # Build abstention head: pooled embedding → abstention rate [0,1]
        # Uses mean pooling over step embeddings, then MLP + sigmoid
        self.abstention_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward_with_abstention(self, L_i: List[Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with abstention: convert log sequence to step probability distribution and abstention rate.
        
        Args:
            L_i: Log sequence (list of steps [L_i[0], ..., L_i[T]])
                 Each step can be a dict or string
        
        Returns:
            p_step: 1D tensor of probabilities [num_steps]
            logits: 1D tensor of raw scores [num_steps]
            a: Scalar abstention rate in [0, 1]
        """
        # 1: Encode all steps into embeddings
        # Input: list of steps
        # Output: [num_steps, d_model]
        step_embeddings = self.encoder(L_i)
        
        # 2: Score each step with a logit (raw score)
        # Input: [num_steps, d_model]
        # Output: [num_steps]
        logits = self.scorer(step_embeddings)
        
        # 3: Convert logits → probability distribution using softmax
        # Input: [num_steps]
        # Output: [num_steps]
        p_step = torch.softmax(logits, dim=0)
        
        # 4: Compute abstention rate from pooled embedding
        # Mean pool over steps: [num_steps, d_model] → [d_model]
        if step_embeddings.shape[0] > 0:
            pooled_embedding = step_embeddings.mean(dim=0)  # [d_model]
        else:
            pooled_embedding = torch.zeros(step_embeddings.shape[1], device=step_embeddings.device)
        
        # Pass through abstention head: [d_model] → [1] → scalar
        a = self.abstention_head(pooled_embedding).squeeze()  # scalar in [0, 1]
        
        return p_step, logits, a
    
    def forward(self, L_i: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: convert log sequence to step probability distribution.
        
        Backward-compatible wrapper around forward_with_abstention (ignores abstention).
        
        Args:
            L_i: Log sequence (list of steps [L_i[0], ..., L_i[T]])
                 Each step can be a dict or string
        
        Returns:
            p_step: 1D tensor of probabilities [num_steps]
            logits: 1D tensor of raw scores [num_steps]
        """
        p_step, logits, _ = self.forward_with_abstention(L_i)
        return p_step, logits
    
    def predict_step_distribution(self, log_steps: List[Any]) -> torch.Tensor:
        """
        Inference wrapper: predict step probability distribution.
        
        Args:
            log_steps: Log sequence (list of steps)
        
        Returns:
            p_step: 1D tensor of probabilities [num_steps]
        """
        # Set to evaluation mode
        self.eval()
        
        with torch.no_grad():
            p_step, _ = self.forward(log_steps)
        
        return p_step


def initialize_discriminators(K: int, model_config: Optional[Dict[str, Any]] = None) -> List[Discriminator]:
    """
    Initialize K discriminators with the given configuration.
    
    Args:
        K: Number of discriminators to create
        model_config: Configuration dictionary for discriminators
                     (see Discriminator.__init__ for details)
    
    Returns:
        List of K Discriminator instances [D_1, ..., D_K]
    """
    discriminators = []
    
    for k in range(1, K + 1):
        D_k = Discriminator(model_config)
        discriminators.append(D_k)
    
    return discriminators

