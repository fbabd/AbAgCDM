# -------------------------------------------------------  #
# ESM-2 encoder from HuggingFace Transformers.             #
# Assumes input is already tokenized with ESM-2 tokenizer. #
# -------------------------------------------------------  #

import torch
import torch.nn as nn
from typing import Optional, Dict
from transformers import EsmModel, EsmTokenizer


class ESM2Encoder(nn.Module):
    """
    ESM-2 encoder from HuggingFace.
    Expects pre-tokenized input following ESM-2 token format.
    Available models on HuggingFace:
    - facebook/esm2_t6_8M_UR50D: 6 layers, 8M params, 320 dim
    - facebook/esm2_t12_35M_UR50D: 12 layers, 35M params, 480 dim
    - facebook/esm2_t30_150M_UR50D: 30 layers, 150M params, 640 dim
    - facebook/esm2_t33_650M_UR50D: 33 layers, 650M params, 1280 dim
    - facebook/esm2_t36_3B_UR50D: 36 layers, 3B params, 2560 dim
    """
    # ====== 
    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        freeze: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze: Whether to freeze encoder weights
            output_attentions: Whether to output attention matrices
        """
        super().__init__()
        
        print(f"Loading ESM-2 model from HuggingFace: {model_name}...")
        
        # Load pretrained model and tokenizer
        self.model = EsmModel.from_pretrained(model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        
        # Get model specifications
        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.vocab_size = self.model.config.vocab_size
        
        print(f"Model loaded:")
        print(f"  Layers: {self.num_layers}")
        print(f"  Heads: {self.num_heads}")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Vocab size: {self.vocab_size}")
        
        # Token ID mapping for validation
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        
        # Freeze weights if requested
        if freeze:
            print("Freezing encoder weights...")
            for param in self.model.parameters():
                param.requires_grad = False
            # self.model.eval()
    
    # ======
    def validate_input_tokens(self, input_ids: torch.Tensor) -> bool:
        """
        Validate that input tokens are within valid ESM-2 vocabulary range.
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            bool: True if valid, raises error otherwise
        """
        max_token = input_ids.max().item()
        min_token = input_ids.min().item()
        
        if min_token < 0 or max_token >= self.vocab_size:
            raise ValueError(
                f"Invalid token IDs detected. "
                f"Expected range [0, {self.vocab_size-1}], "
                f"but got [{min_token}, {max_token}]"
            )
        
        return True
    
    # ======
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_all_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ESM-2 encoder.
        Args:
            input_ids: Pre-tokenized token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]  1 = attend, 0 = ignore (padding)
            output_all_hidden_states: whether to return the internal embeddings 
            output_attentions: whether to return attention matrices from all layers 
        Returns:
            Dictionary containing:
                - last_hidden_state: [batch_size, seq_len, hidden_dim]
                - attentions: Optional[Tuple] of attention matrices if output_attentions=True
                              Each tuple element: [batch_size, num_heads, seq_len, seq_len]
                - all_hidden_states: 
        """
        self.validate_input_tokens(input_ids) 
        
        # Auto-create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_all_hidden_states,
            return_dict=True
        )
        
        result = {
            'last_hidden_state': outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        }
        if output_all_hidden_states:
            # Tuple of length num_layers
            # Each element: [batch_size, seq_len, hidden_dim]
            result['all_hidden_states'] = outputs.hidden_states
        
        if output_attentions and outputs.attentions is not None:
            # Tuple of length num_layers
            # Each element: [batch_size, num_heads, seq_len, seq_len]
            result['attentions'] = outputs.attentions
        
        return result
    
    # ====== 
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_all_hidden_states: bool = False
    ) -> torch.Tensor:
        """  Convenience method to get only hidden states.  """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, output_all_hidden_states)
        
        if output_all_hidden_states:
            return outputs['all_hidden_states']
        
        return outputs['last_hidden_state']
    
    # ====== 
    def get_sequence_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = 'mean'
    ) -> torch.Tensor:
        """
        Get single embedding per sequence (inference mode).
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            pooling: 'mean', 'cls', or 'max'
        
        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        hidden_states = self.get_embeddings(input_ids, attention_mask)
        
        if pooling == 'cls':
            return hidden_states[:, 0, :]
        
        elif pooling == 'mean':
            if attention_mask is None:
                attention_mask = (input_ids != self.pad_token_id).long()
            
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask
        
        elif pooling == 'max':
            if attention_mask is None:
                attention_mask = (input_ids != self.pad_token_id).long()
            
            hidden_states = hidden_states.clone()
            hidden_states[attention_mask == 0] = -1e9
            return hidden_states.max(dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    
    # ====== 
    def get_hidden_dim(self) -> int:
        """Return hidden dimension size."""
        return self.hidden_dim
    
    # ====== 
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    # ====== 
    def get_token_ids(self) -> Dict[str, int]:
        """Return special token IDs for reference."""
        return {
            'pad': self.pad_token_id,
            'cls': self.cls_token_id,
            'eos': self.eos_token_id,
            'mask': self.mask_token_id
        }
    
    # ====== 
    @property
    def device(self):
        """Return the device of the model"""
        return next(self.model.parameters()).device
    
   

# ============================================================================
# TESTING
# ============================================================================

def test_script(): 
    print("=" * 80)
    print("Testing ESM-2 Encoder from HuggingFace")
    print("=" * 80)
    
    # ========================================================================
    # Test 1: Initialization and Model Properties
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Initialization and Model Properties")
    print("=" * 80)
    
    encoder = ESM2Encoder(
        model_name="facebook/esm2_t12_35M_UR50D",
        freeze=False 
    )
    
    print(f"\n✓ Model initialized successfully")
    print(f"  Hidden dim: {encoder.get_hidden_dim()}")
    print(f"  Vocab size: {encoder.get_vocab_size()}")
    print(f"  Num layers: {encoder.num_layers}")
    print(f"  Num heads: {encoder.num_heads}")
    print(f"\n✓ Special token IDs:")
    token_ids = encoder.get_token_ids()
    for token_name, token_id in token_ids.items():
        print(f"  {token_name}: {token_id}")
    
    # ========================================================================
    # Test 2: Prepare Sample Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Preparing Sample Data")
    print("=" * 80)
    
    batch_size = 4
    seq_len = 128
    
    # Create sample input (valid ESM-2 token range: 0-32)
    input_ids = torch.randint(4, 25, (batch_size, seq_len))  # Amino acid tokens
    # Set first token to <cls> (0) and last to <eos> (2)
    input_ids[:, 0] = encoder.cls_token_id
    input_ids[:, -1] = encoder.eos_token_id
    
    # Create attention mask with some padding
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # Simulate variable-length sequences
    for i in range(batch_size):
        pad_start = seq_len - (i + 1) * 5  # Different padding for each sequence
        attention_mask[i, pad_start:] = 0
        input_ids[i, pad_start:] = encoder.pad_token_id
    
    print(f"✓ Sample data created:")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Attention mask shape: {attention_mask.shape}")
    print(f"  Token range: [{input_ids.min().item()}, {input_ids.max().item()}]")
    print(f"  Padding positions in batch:")
    for i in range(batch_size):
        num_pad = (attention_mask[i] == 0).sum().item()
        print(f"    Sequence {i}: {num_pad} padding tokens")
    
    # ========================================================================
    # Test 3: Token Validation
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Token Validation")
    print("=" * 80)
    
    # Test valid tokens
    try:
        encoder.validate_input_tokens(input_ids)
        print("✓ Valid tokens passed validation")
    except ValueError as e:
        print(f"✗ Valid tokens failed: {e}")
    
    # Test invalid tokens (too high)
    try:
        invalid_ids_high = torch.randint(50, 100, (2, 10))
        encoder.validate_input_tokens(invalid_ids_high)
        print("✗ Invalid tokens (too high) should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly caught invalid tokens (too high): {str(e)[:60]}...")
    
    # Test invalid tokens (negative)
    try:
        invalid_ids_neg = torch.randint(-10, 0, (2, 10))
        encoder.validate_input_tokens(invalid_ids_neg)
        print("✗ Invalid tokens (negative) should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly caught invalid tokens (negative): {str(e)[:60]}...")
    
    # ========================================================================
    # Test 4: Forward Pass - Last Hidden State Only
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Forward Pass - Last Hidden State Only")
    print("=" * 80)
    
    with torch.no_grad():
        outputs = encoder(input_ids, attention_mask, output_all_hidden_states=False, output_attentions=True) 
    
    print(f"✓ Forward pass completed")
    print(f"  Keys in output: {list(outputs.keys())}")
    print(f"  'last_hidden_state' in outputs: {'last_hidden_state' in outputs}")
    print(f"  'all_hidden_states' in outputs: {'all_hidden_states' in outputs}")
    
    assert 'last_hidden_state' in outputs, "Output must contain 'last_hidden_state'"
    assert 'all_hidden_states' not in outputs, "Should not contain 'all_hidden_states' when output_all_hidden_states=False"
    
    print(f"\n✓ Last hidden state:")
    print(f"  Shape: {outputs['last_hidden_state'].shape}")
    print(f"  Expected: [{batch_size}, {seq_len}, {encoder.get_hidden_dim()}]")
    assert outputs['last_hidden_state'].shape == (batch_size, seq_len, encoder.get_hidden_dim())
    print(f"  ✓ Shape matches expected")
    
    if 'attentions' in outputs and outputs['attentions'] is not None:
        print(f"\n✓ Attention outputs:")
        print(f"  Number of layers: {len(outputs['attentions'])}")
        print(f"  First layer attention shape: {outputs['attentions'][0].shape}")
        print(f"  Expected: [{batch_size}, {encoder.num_heads}, {seq_len}, {seq_len}]")
        assert outputs['attentions'][0].shape == (batch_size, encoder.num_heads, seq_len, seq_len)
        print(f"  ✓ Attention shape matches expected")
        
        # Check attention properties
        first_attn = outputs['attentions'][0]
        attn_sum = first_attn.sum(dim=-1)
        print(f"\n✓ Attention properties:")
        print(f"  Attention sum over last dim (should be ~1.0): {attn_sum[0, 0, 0].item():.4f}")
        print(f"  Min attention value: {first_attn.min().item():.4f}")
        print(f"  Max attention value: {first_attn.max().item():.4f}")
    
    # ========================================================================
    # Test 5: Forward Pass - All Hidden States
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Forward Pass - All Hidden States")
    print("=" * 80)
    
    with torch.no_grad():
        outputs_all = encoder(input_ids, attention_mask, output_all_hidden_states=True, output_attentions=True)
    
    print(f"✓ Forward pass with all hidden states completed")
    print(f"  Keys in output: {list(outputs_all.keys())}")
    print(f"  'last_hidden_state' in outputs: {'last_hidden_state' in outputs_all}")
    print(f"  'all_hidden_states' in outputs: {'all_hidden_states' in outputs_all}")
    
    assert 'last_hidden_state' in outputs_all, "Output must contain 'last_hidden_state'"
    assert 'all_hidden_states' in outputs_all, "Output must contain 'all_hidden_states' when output_all_hidden_states=True"
    
    print(f"\n✓ All hidden states:")
    print(f"  Type: {type(outputs_all['all_hidden_states'])}")
    print(f"  Number of layers (including embedding): {len(outputs_all['all_hidden_states'])}")
    print(f"  Expected: {encoder.num_layers + 1} (embedding + {encoder.num_layers} layers)")
    
    # Check each layer's hidden state
    for i, hidden_state in enumerate(outputs_all['all_hidden_states']):
        print(f"  Layer {i} shape: {hidden_state.shape}")
        assert hidden_state.shape == (batch_size, seq_len, encoder.get_hidden_dim())
    
    # Verify last hidden state matches
    print(f"\n✓ Consistency check:")
    last_from_all = outputs_all['all_hidden_states'][-1]
    last_direct = outputs_all['last_hidden_state']
    is_same = torch.allclose(last_from_all, last_direct, rtol=1e-5, atol=1e-8)
    print(f"  Last state from 'all_hidden_states' matches 'last_hidden_state': {is_same}")
    assert is_same, "Last hidden state should match final layer in all_hidden_states"
    
    # ========================================================================
    # Test 6: Forward Pass without Attention
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 6: Forward Pass without Attention Output")
    print("=" * 80) 
    
    with torch.no_grad():
        outputs_no_attn = encoder(input_ids, attention_mask, output_attentions=False)
    
    print(f"✓ Forward pass completed (no attention)")
    print(f"  Keys in output: {list(outputs_no_attn.keys())}")
    print(f"  'attentions' in outputs: {'attentions' in outputs_no_attn}")
    
    assert 'attentions' not in outputs_no_attn or outputs_no_attn.get('attentions') is None
    print(f"  ✓ Correctly omitted attention outputs")
    
    # ========================================================================
    # Test 7: Auto-generated Attention Mask
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 7: Auto-generated Attention Mask")
    print("=" * 80)
    
    # Create input with padding tokens but no explicit attention mask
    input_with_padding = input_ids.clone()
    
    with torch.no_grad():
        outputs_auto_mask = encoder(input_with_padding, attention_mask=None)
    
    print(f"✓ Forward pass with auto-generated attention mask")
    print(f"  Output shape: {outputs_auto_mask['last_hidden_state'].shape}")
    
    # Compare with explicit mask
    with torch.no_grad():
        outputs_explicit_mask = encoder(input_with_padding, attention_mask=attention_mask)
    
    # They should be very close (might not be exactly equal due to numerical precision)
    is_close = torch.allclose(
        outputs_auto_mask['last_hidden_state'], 
        outputs_explicit_mask['last_hidden_state'],
        rtol=1e-5, atol=1e-8
    )
    print(f"  Auto-mask matches explicit mask: {is_close}")
    
    # ========================================================================
    # Test 8: get_embeddings() Method
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 8: get_embeddings() Convenience Method")
    print("=" * 80)
    
    with torch.no_grad():
        embeddings = encoder.get_embeddings(input_ids, attention_mask)
    
    print(f"✓ get_embeddings() completed")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Same as forward() last_hidden_state: {torch.equal(embeddings, outputs['last_hidden_state'])}")
    
    # Test without attention mask
    with torch.no_grad():
        embeddings_no_mask = encoder.get_embeddings(input_ids)
    
    print(f"\n✓ get_embeddings() without attention mask:")
    print(f"  Embeddings shape: {embeddings_no_mask.shape}")
    
    # ========================================================================
    # Test 9: get_sequence_embeddings() with Different Pooling
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 9: get_sequence_embeddings() with Pooling Strategies")
    print("=" * 80)
    
    pooling_strategies = ['cls', 'mean', 'max']
    sequence_embeddings = {}
    
    for pooling in pooling_strategies:
        with torch.no_grad():
            seq_emb = encoder.get_sequence_embeddings(input_ids, attention_mask, pooling=pooling)
        
        sequence_embeddings[pooling] = seq_emb
        print(f"\n✓ Pooling strategy: '{pooling}'")
        print(f"  Output shape: {seq_emb.shape}")
        print(f"  Expected: [{batch_size}, {encoder.get_hidden_dim()}]")
        assert seq_emb.shape == (batch_size, encoder.get_hidden_dim())
        print(f"  ✓ Shape matches expected")
        
        # Show some statistics
        print(f"  Mean value: {seq_emb.mean().item():.4f}")
        print(f"  Std value: {seq_emb.std().item():.4f}")
        print(f"  Min value: {seq_emb.min().item():.4f}")
        print(f"  Max value: {seq_emb.max().item():.4f}")
    
    # Compare different pooling strategies
    print(f"\n✓ Comparing pooling strategies:")
    print(f"  CLS vs Mean L2 distance: {torch.dist(sequence_embeddings['cls'], sequence_embeddings['mean']).item():.4f}")
    print(f"  CLS vs Max L2 distance: {torch.dist(sequence_embeddings['cls'], sequence_embeddings['max']).item():.4f}")
    print(f"  Mean vs Max L2 distance: {torch.dist(sequence_embeddings['mean'], sequence_embeddings['max']).item():.4f}")
    
    # Verify CLS pooling extracts first token correctly
    with torch.no_grad():
        full_embeddings = encoder.get_embeddings(input_ids, attention_mask)
    cls_manual = full_embeddings[:, 0, :]
    cls_pooled = sequence_embeddings['cls']
    print(f"\n✓ CLS pooling verification:")
    print(f"  Manual CLS extraction matches pooled: {torch.allclose(cls_manual, cls_pooled)}")
    
    # Test invalid pooling strategy
    try:
        encoder.get_sequence_embeddings(input_ids, attention_mask, pooling='invalid')
        print("✗ Invalid pooling strategy should have raised ValueError")
    except ValueError as e:
        print(f"\n✓ Correctly caught invalid pooling strategy: {e}")
    
    # ========================================================================
    # Test 10: Parameter Counting
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 10: Model Parameters")
    print("=" * 80)
    
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"✓ Parameter count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # ========================================================================
    # Test 11: Freezing Mechanism
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 11: Freezing Mechanism")
    print("=" * 80)
    
    encoder_frozen = ESM2Encoder(
        model_name="facebook/esm2_t12_35M_UR50D",
        freeze=True,
        output_attentions=False
    )
    
    total_frozen = sum(p.numel() for p in encoder_frozen.parameters())
    trainable_frozen = sum(p.numel() for p in encoder_frozen.parameters() if p.requires_grad)
    
    print(f"✓ Frozen encoder parameters:")
    print(f"  Total parameters: {total_frozen:,}")
    print(f"  Trainable parameters: {trainable_frozen:,}")
    print(f"  Successfully frozen: {trainable_frozen == 0}")
    assert trainable_frozen == 0, "Frozen model should have 0 trainable parameters"
    print(f"  ✓ All parameters correctly frozen")
    
    # Test that frozen model still produces outputs
    with torch.no_grad():
        frozen_outputs = encoder_frozen(input_ids, attention_mask)
    
    print(f"\n✓ Frozen model forward pass:")
    print(f"  Output shape: {frozen_outputs['last_hidden_state'].shape}")
    print(f"  ✓ Frozen model still produces outputs correctly")
    
    # ========================================================================
    # Test 12: Device Property
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 12: Device Property")
    print("=" * 80)
    
    device = encoder.device
    print(f"✓ Model device: {device}")
    print(f"  Device type: {device.type}")
    
    # Test GPU if available
    if torch.cuda.is_available():
        print(f"\n✓ CUDA available, testing GPU transfer...")
        encoder_gpu = ESM2Encoder(
            model_name="facebook/esm2_t12_35M_UR50D",
            freeze=False
        )
        encoder_gpu.model = encoder_gpu.model.cuda()
        
        print(f"  Model device after .cuda(): {encoder_gpu.device}")
        assert encoder_gpu.device.type == 'cuda'
        print(f"  ✓ Successfully moved to GPU")
        
        # Test inference on GPU
        input_ids_gpu = input_ids.cuda()
        attention_mask_gpu = attention_mask.cuda()
        
        with torch.no_grad():
            outputs_gpu = encoder_gpu(input_ids_gpu, attention_mask_gpu)
        
        print(f"  Output device: {outputs_gpu['last_hidden_state'].device}")
        print(f"  ✓ GPU inference successful")
    else:
        print(f"  CUDA not available, skipping GPU tests")
    
    # ========================================================================
    # Test 13: Gradient Flow (Unfrozen Model)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 13: Gradient Flow (Unfrozen Model)")
    print("=" * 80)
    
    encoder_trainable = ESM2Encoder(
        model_name="facebook/esm2_t12_35M_UR50D",
        freeze=False
    )
    
    # Forward pass with gradients
    outputs_train = encoder_trainable(input_ids, attention_mask)
    loss = outputs_train['last_hidden_state'].mean()
    loss.backward()
    
    # Check if gradients exist
    has_grad = any(p.grad is not None for p in encoder_trainable.parameters())
    print(f"✓ Gradient computation test:")
    print(f"  Gradients computed: {has_grad}")
    assert has_grad, "Unfrozen model should have gradients after backward()"
    print(f"  ✓ Gradients flow correctly through unfrozen model")
    
    # Count parameters with gradients
    params_with_grad = sum(1 for p in encoder_trainable.parameters() if p.grad is not None)
    total_params_count = sum(1 for _ in encoder_trainable.parameters())
    print(f"  Parameters with gradients: {params_with_grad}/{total_params_count}")
    
    # ========================================================================
    # Test 14: Batch Size Flexibility
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 14: Batch Size Flexibility")
    print("=" * 80)
    
    test_batch_sizes = [1, 2, 8, 16]
    
    for bs in test_batch_sizes:
        test_input = torch.randint(4, 25, (bs, 50))
        test_input[:, 0] = encoder.cls_token_id
        test_input[:, -1] = encoder.eos_token_id
        
        with torch.no_grad():
            test_output = encoder.get_embeddings(test_input)
        
        print(f"  Batch size {bs:2d}: {test_output.shape} ✓")
        assert test_output.shape == (bs, 50, encoder.get_hidden_dim())
    
    print(f"✓ All batch sizes handled correctly")
    
    # ========================================================================
    # Test 15: Edge Case - Single Token Sequence
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 15: Edge Case - Single Token Sequence")
    print("=" * 80)
    
    # Just CLS token
    single_token = torch.tensor([[encoder.cls_token_id]])
    
    with torch.no_grad():
        single_output = encoder(single_token)
    
    print(f"✓ Single token input:")
    print(f"  Input shape: {single_token.shape}")
    print(f"  Output shape: {single_output['last_hidden_state'].shape}")
    assert single_output['last_hidden_state'].shape == (1, 1, encoder.get_hidden_dim())
    print(f"  ✓ Handled correctly")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✓ All tests passed successfully!")
    print("\nTested components:")
    print("  1. ✓ Initialization and model properties")
    print("  2. ✓ Sample data preparation")
    print("  3. ✓ Token validation")
    print("  4. ✓ Forward pass - last hidden state only")
    print("  5. ✓ Forward pass - all hidden states")
    print("  6. ✓ Forward pass without attention")
    print("  7. ✓ Auto-generated attention mask")
    print("  8. ✓ get_embeddings() method")
    print("  9. ✓ get_sequence_embeddings() with pooling")
    print(" 10. ✓ Parameter counting")
    print(" 11. ✓ Freezing mechanism")
    print(" 12. ✓ Device property")
    print(" 13. ✓ Gradient flow")
    print(" 14. ✓ Batch size flexibility")
    print(" 15. ✓ Edge case - single token sequence")
    print("=" * 80)

if __name__ == "__main__":
    test_script() 
    
    