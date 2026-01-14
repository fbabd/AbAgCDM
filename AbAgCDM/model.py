"""
    Multi-task model for antibody-antigen binding prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

try:
    from .encoder_esm2 import ESM2Encoder
except ImportError:
    from encoder_esm2 import ESM2Encoder

# Task A
class BindingHead(nn.Module):
    """
    Task A: Binary binding prediction head. 
    Predicts if VHH binds to IL-6 variant. 
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, seq_embedding: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            seq_embedding (torch.Tensor): [batch_size, hidden_dim] from [CLS] token
        Returns:
            logits (torch.Tensor): [batch_size] binding prediction logits
        """
        logits = self.classifier(seq_embedding).squeeze(-1)
        return logits


# Task B 
class ContrastiveHead(nn.Module):
    """
    Task B: Contrastive learning head for coevolution. 
    Projects VHH and IL-6 embeddings to shared space for similarity learning.
    """
    def __init__(
        self, 
        hidden_dim: int, 
        contrastive_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.contrastive_dim = contrastive_dim
        
        # Separate projectors for VHH and IL-6
        self.vhh_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, contrastive_dim)
        )
        
        self.il6_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, contrastive_dim)
        )
    
    def forward(
        self, 
        vhh_embedding: torch.Tensor, 
        il6_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vhh_embedding: [batch_size, hidden_dim]
            il6_embedding: [batch_size, hidden_dim]
        
        Returns:
            z_vhh: [batch_size, contrastive_dim] normalized
            z_il6: [batch_size, contrastive_dim] normalized
        """
        # Project to contrastive space
        z_vhh = self.vhh_projector(vhh_embedding)
        z_il6 = self.il6_projector(il6_embedding)
        
        # L2 normalize for cosine similarity
        z_vhh = F.normalize(z_vhh, p=2, dim=-1)
        z_il6 = F.normalize(z_il6, p=2, dim=-1)
        
        return z_vhh, z_il6


# custom loss 
class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for positive class
        self.gamma = gamma  # Focusing parameter
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Alpha balancing
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_t * focal_term * bce_loss
        return loss.mean()


# AbAg Contrastive–Discriminative Model (AbAg-CDM) 

class AbAgCDM(nn.Module):     
    def __init__(
        self,
        encoder: ESM2Encoder,
        contrastive_dim: int = 256,
        dropout: float = 0.1,
        temperature: float = 0.07,
        lambda_contrastive: float = 0.33,
        pooling_method: str = "mean",
        use_focal_loss: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        pos_weight: float = 32.0  # 97/3 ratio
    ):
        """
        Args:
            encoder: ESM2Encoder instance
            contrastive_dim: Dimension for contrastive embeddings
            dropout: Dropout rate
            temperature: Temperature for contrastive loss
            lambda_contrastive: Weight for contrastive loss
            pooling_method: How to pool regions ("mean", "max", "cls")
        """ 
        super().__init__()
        
        self.encoder = encoder
        self.hidden_dim = encoder.get_hidden_dim()
        self.contrastive_dim = contrastive_dim
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.pooling_method = pooling_method
        
        # Two task heads
        self.binding_head = BindingHead(self.hidden_dim, dropout)
        self.contrastive_head = ContrastiveHead(self.hidden_dim, contrastive_dim, dropout)
        
        # Loss function 
        self.use_focal_loss = use_focal_loss
        self.pos_weight = torch.tensor([pos_weight])
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma) 
        
        print(f"Model initialized:")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Contrastive dim: {self.contrastive_dim}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Lambda contrastive: {self.lambda_contrastive}")

    
    # ====== 
    def extract_region_embedding(
        self,
        hidden_states: torch.Tensor,
        start_idx: int, # inclusive
        end_idx: int    # exclusive 
    ) -> torch.Tensor:
        """
        Extract embedding for a region (VHH or IL-6).
        
        Args:
            hidden_states: [seq_len, hidden_dim]
            start_idx: Start index
            end_idx: End index
        
        Returns:
            embedding: [hidden_dim]
        """
        region = hidden_states[start_idx:end_idx, :]
        
        if self.pooling_method == "mean":
            return region.mean(dim=0)
        elif self.pooling_method == "max":
            return region.max(dim=0)[0]
        elif self.pooling_method == "first":
            return region[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
    
    # ====== 
    def compute_binding_loss(
        self,
        encoder_output: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_output: [batch_size, seq_len, hidden_dim]
            labels: [batch_size] binary labels
        Returns:
            loss: scalar
            logits: [batch_size]
        """
        # Use [CLS] token (position 0)
        cls_embedding = encoder_output[:, 0, :]  # [batch_size, hidden_dim] 
        # Get binding logits
        logits = self.binding_head(cls_embedding)  # [batch_size]
        
        if self.use_focal_loss:
            loss = self.focal_loss(logits, labels.float())
        else:
            # Weighted BCE
            self.pos_weight = self.pos_weight.to(logits.device)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(),
                pos_weight=self.pos_weight
            ) 
        
        return loss, logits
    
    # ====== 
    def compute_contrastive_loss(
        self,
        encoder_output: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        Args:
            encoder_output: [batch_size, seq_len, hidden_dim]
            batch: Batch dictionary with negative examples
        
        Returns:
            loss: scalar
            z_vhh: [batch_size, contrastive_dim]
            z_il6: [batch_size, contrastive_dim]
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device
        
        # Extract VHH and IL-6 embeddings for POSITIVE examples
        vhh_embeddings = []
        il6_embeddings = []
        
        for i in range(batch_size):
            il6_start = batch['il6_start_idx'][i].item()
            il6_end = batch['il6_end_idx'][i].item()
            # VHH: after [CLS] (pos 1), before first [SEP]
            vhh_emb = self.extract_region_embedding(encoder_output[i], 1, il6_start - 1) 
            # IL-6: between [SEP] tokens
            il6_emb = self.extract_region_embedding(encoder_output[i], il6_start, il6_end) 
            vhh_embeddings.append(vhh_emb)
            il6_embeddings.append(il6_emb)
        
        vhh_embeddings = torch.stack(vhh_embeddings)  # [batch_size, hidden_dim]
        il6_embeddings = torch.stack(il6_embeddings)  # [batch_size, hidden_dim]
        
        # Project to contrastive space
        z_vhh_pos, z_il6_pos = self.contrastive_head(vhh_embeddings, il6_embeddings)
        # z_vhh_pos, z_il6_pos = vhh_embeddings, il6_embeddings
        
        
        
        # Compute contrastive loss with hard negatives
        loss_contrastive = torch.tensor(0.0, device=device)
        num_contrastive = 0
        
        for i in range(batch_size):
            # Only compute for examples with negatives
            if not batch['has_negatives'][i]:
                continue
            
            num_neg = batch['num_negatives'][i].item()
            if num_neg == 0:
                continue
            
            # Encode negative examples
            neg_input_ids = batch['negative_input_ids'][i, :num_neg, :]
            neg_attention_mask = batch['negative_attention_mask'][i, :num_neg, :]
            
            # Forward pass for negatives
            neg_outputs = self.encoder(neg_input_ids, neg_attention_mask)
            neg_hidden = neg_outputs['last_hidden_state']  # [num_neg, seq_len, hidden_dim]
            
            # Extract IL-6 embeddings from negatives
            il6_start = batch['il6_start_idx'][i].item()
            il6_end = batch['il6_end_idx'][i].item()
            
            # Extract BOTH VHH and AG from NEGATIVE samples 
            vhh_neg_embeddings = []
            il6_neg_embeddings = []
            for j in range(num_neg):
                # Extract VHH from this negative encoding
                vhh_neg_emb = self.extract_region_embedding(neg_hidden[j], 1, il6_start - 1)
                vhh_neg_embeddings.append(vhh_neg_emb)
                
                # Extract AG from this negative encoding  
                il6_neg_emb = self.extract_region_embedding(neg_hidden[j], il6_start, il6_end)
                il6_neg_embeddings.append(il6_neg_emb)
            
            vhh_neg_embeddings = torch.stack(vhh_neg_embeddings)  # [num_neg, hidden_dim]
            il6_neg_embeddings = torch.stack(il6_neg_embeddings)  # [num_neg, hidden_dim]
            
            # Project to contrastive space 
            z_vhh_neg, z_il6_neg = self.contrastive_head(vhh_neg_embeddings, il6_neg_embeddings)
            # z_vhh_neg, z_il6_neg = vhh_neg_embeddings, il6_neg_embeddings
        
            
            # InfoNCE with interaction modeling
            #   Option A: Similarity in joint embedding space
            pos_interaction = torch.sum(z_vhh_pos[i] * z_il6_pos[i])  # Scalar
            neg_interactions = torch.sum(z_vhh_neg * z_il6_neg, dim=1)  # [num_neg]
            #   Option B: Cross-similarity (old)
            # pos_interaction = torch.sum(z_vhh_pos[i] * z_il6_pos[i])
            # neg_interactions = torch.matmul(z_vhh_pos[i].unsqueeze(0), z_il6_neg.T).squeeze(0)
            logits = torch.cat([pos_interaction.unsqueeze(0), neg_interactions]) / self.temperature
            target = torch.zeros(1, dtype=torch.long, device=device)
            loss_contrastive += F.cross_entropy(logits.unsqueeze(0), target)
            num_contrastive += 1
        
        # Average over examples with negatives
        if num_contrastive > 0:
            loss_contrastive = loss_contrastive / num_contrastive
        
        return loss_contrastive, z_vhh_pos, z_il6_pos
    

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dictionary from DataLoader containing:
                - input_ids: [batch_size, seq_len]
                - attention_mask: [batch_size, seq_len]
                - il6_start_idx: [batch_size]
                - il6_end_idx: [batch_size]
                - binding_label: [batch_size]
                - has_negatives: [batch_size]
                - negative_input_ids: [batch_size, num_neg, seq_len]
                - negative_attention_mask: [batch_size, num_neg, seq_len]
                - num_negatives: [batch_size] 
        
        Returns:
            Dictionary with losses and predictions
        """
        # ===================================================================
        # SHARED ENCODER
        # ===================================================================
        encoder_outputs = self.encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        last_hidden_state = encoder_outputs['last_hidden_state']  # [batch_size, seq_len, hidden_dim]
        
        # ===================================================================
        # TASK A: BINDING PREDICTION
        # ===================================================================
        loss_binding, binding_logits = self.compute_binding_loss(
            last_hidden_state, batch['binding_label']
        )
        
        # ===================================================================
        # TASK B: CONTRASTIVE LEARNING
        # ===================================================================
        loss_contrastive, z_vhh, z_il6 = self.compute_contrastive_loss(
            last_hidden_state, batch
        )
        

        # ===================================================================
        # COMBINED LOSS
        # ===================================================================
        total_loss = (
            (1 - self.lambda_contrastive) * loss_binding +
            self.lambda_contrastive * loss_contrastive 
        )
        
        # ===================================================================
        # RETURN OUTPUTS
        # ===================================================================
        return {
            'loss': total_loss,
            'loss_binding': loss_binding.item(),
            'loss_contrastive': loss_contrastive.item(),
            'binding_logits': binding_logits,
            'binding_probs': torch.sigmoid(binding_logits),
            'z_vhh': z_vhh,
            'z_il6': z_il6,
            'encoder_output': last_hidden_state
        }
    
    @torch.no_grad()
    def predict_binding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            probs: [batch_size] binding probabilities
        """
        self.eval()
        encoder_outputs = self.encoder(input_ids, attention_mask)
        hidden_states = encoder_outputs['last_hidden_state']
        cls_embedding = hidden_states[:, 0, :]
        logits = self.binding_head(cls_embedding)
        probs = torch.sigmoid(logits)
        return probs
    

    @torch.no_grad()
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        il6_start_idx: torch.Tensor,   # [B]
        il6_end_idx: torch.Tensor,     # [B] (typically points to EOS right after IL6)
        attention_mask: Optional[torch.Tensor] = None,
        return_token_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings for CLS, VHH region, and IL6 region from a concatenated input:
        [CLS] VHH ... [EOS] IL6 ... [EOS]

        Args:
            input_ids: [B, L]
            il6_start_idx: [B] start index of IL6 tokens (first IL6 token)
            il6_end_idx:   [B] end index (typically the EOS after IL6). Treated as end-exclusive.
            attention_mask: [B, L]
            return_token_embeddings: if True, returns per-token embeddings [B, L, H]

        Returns:
            dict with:
            - "cls": [B, H]
            - "vhh": [B, H]
            - "il6": [B, H]
            - optionally "token": [B, L, H]
        """
        self.eval() 
        enc = self.encoder(input_ids, attention_mask)
        hidden_states = enc["last_hidden_state"]  # [B, L, H]

        B, L, H = hidden_states.shape
        out: Dict[str, torch.Tensor] = {}

        # CLS
        out["cls"] = hidden_states[:, 0, :]  # [B, H]

        # Ensure indices are on CPU ints for python slicing
        il6_start_idx_cpu = il6_start_idx.to("cpu").tolist()
        il6_end_idx_cpu = il6_end_idx.to("cpu").tolist()

        vhh_embs = []
        il6_embs = []

        for b in range(B):
            il6_s = int(il6_start_idx_cpu[b])
            il6_e = int(il6_end_idx_cpu[b])

            # VHH is right after CLS, ends right before the EOS that precedes IL6
            vhh_s = 1
            vhh_eos = il6_s - 1          # index of the EOS between VHH and IL6
            vhh_e = vhh_eos              # end-exclusive => excludes that EOS

            # Basic sanity checks (helps catch off-by-ones early)
            if not (0 <= vhh_s <= vhh_e <= L):
                raise ValueError(f"[sample {b}] Bad VHH span computed: ({vhh_s}, {vhh_e}) with L={L}")
            if not (0 <= il6_s <= il6_e <= L):
                raise ValueError(f"[sample {b}] Bad IL6 span: ({il6_s}, {il6_e}) with L={L}")
            if vhh_e <= vhh_s:
                raise ValueError(f"[sample {b}] Empty VHH span: ({vhh_s}, {vhh_e})")
            if il6_e <= il6_s:
                raise ValueError(f"[sample {b}] Empty IL6 span: ({il6_s}, {il6_e})")

            # hidden_states[b] is [L, H], matches your helper
            vhh_embs.append(self.extract_region_embedding(hidden_states[b], vhh_s, vhh_e))
            il6_embs.append(self.extract_region_embedding(hidden_states[b], il6_s, il6_e))

        out["vhh"] = torch.stack(vhh_embs, dim=0)  # [B, H]
        out["il6"] = torch.stack(il6_embs, dim=0)  # [B, H]

        if return_token_embeddings:
            out["token"] = hidden_states  # [B, L, H]

        return out


    @torch.no_grad()
    def get_contrastive_embeddings(
        self,
        input_ids: torch.Tensor,
        il6_start_idx: torch.Tensor,   # [B]
        il6_end_idx: torch.Tensor,     # [B] (typically points to EOS right after IL6)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings for CLS, VHH region, and IL6 region from a concatenated input:
        [CLS] VHH ... [EOS] IL6 ... [EOS]

        Args:
            input_ids: [B, L]
            il6_start_idx: [B] start index of IL6 tokens (first IL6 token)
            il6_end_idx:   [B] end index (typically the EOS after IL6). Treated as end-exclusive.
            attention_mask: [B, L]
            return_token_embeddings: if True, returns per-token embeddings [B, L, H]

        Returns:
            dict with:
            - "cls": [B, H]
            - "vhh": [B, H]
            - "il6": [B, H]
            - optionally "token": [B, L, H]
        """
        self.eval() 
        enc = self.encoder(input_ids, attention_mask)
        hidden_states = enc["last_hidden_state"]  # [B, L, H]
        
        B, L, H = hidden_states.shape
        out: Dict[str, torch.Tensor] = {}

        # CLS
        out["cls"] = hidden_states[:, 0, :]  # [B, H]

        # Ensure indices are on CPU ints for python slicing
        il6_start_idx_cpu = il6_start_idx.to("cpu").tolist()
        il6_end_idx_cpu = il6_end_idx.to("cpu").tolist()

        vhh_embs = []
        il6_embs = []

        for b in range(B):
            il6_s = int(il6_start_idx_cpu[b])
            il6_e = int(il6_end_idx_cpu[b])

            # VHH is right after CLS, ends right before the EOS that precedes IL6
            vhh_s = 1
            vhh_eos = il6_s - 1          # index of the EOS between VHH and IL6
            vhh_e = vhh_eos              # end-exclusive => excludes that EOS

            # Basic sanity checks (helps catch off-by-ones early)
            if not (0 <= vhh_s <= vhh_e <= L):
                raise ValueError(f"[sample {b}] Bad VHH span computed: ({vhh_s}, {vhh_e}) with L={L}")
            if not (0 <= il6_s <= il6_e <= L):
                raise ValueError(f"[sample {b}] Bad IL6 span: ({il6_s}, {il6_e}) with L={L}")
            if vhh_e <= vhh_s:
                raise ValueError(f"[sample {b}] Empty VHH span: ({vhh_s}, {vhh_e})")
            if il6_e <= il6_s:
                raise ValueError(f"[sample {b}] Empty IL6 span: ({il6_s}, {il6_e})")

            # hidden_states[b] is [L, H], matches your helper
            vhh_embs.append(self.extract_region_embedding(hidden_states[b], vhh_s, vhh_e))
            il6_embs.append(self.extract_region_embedding(hidden_states[b], il6_s, il6_e))

        out["vhh"] = torch.stack(vhh_embs, dim=0)  # [B, H]
        out["il6"] = torch.stack(il6_embs, dim=0)  # [B, H]
        
        out["vhh"], out["il6"] = self.contrastive_head(out["vhh"], out["il6"])
        
        return out 
        
        
        
# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(
    encoder_name: str = "facebook/esm2_t12_35M_UR50D",
    freeze_encoder: bool = False,
    contrastive_dim: int = 256,
    dropout: float = 0.1,
    temperature: float = 0.07,
    lambda_contrastive: float = 0.5,
    pooling_method: str = "mean",
    use_focal_loss: bool = True,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    pos_weight: float = 32.0  # 97/3 ratio
) -> AbAgCDM:
    """
    Factory function to create the model.
    
    Args:
        encoder_name: HuggingFace ESM-2 model name
        freeze_encoder: Whether to freeze encoder
        contrastive_dim: Contrastive embedding dimension
        dropout: Dropout rate
        temperature: Contrastive loss temperature
        lambda_contrastive: Weight for contrastive loss
        pooling_method: Region pooling method
        output_attentions: Whether to output attention matrices
    
    Returns:
        CoevolutionBindingModel instance
    """
    # Create encoder
    encoder = ESM2Encoder(
        model_name=encoder_name,
        freeze=freeze_encoder,
    )
    
    # Create model
    model = AbAgCDM(
        encoder=encoder,
        contrastive_dim=contrastive_dim,
        dropout=dropout,
        temperature=temperature,
        lambda_contrastive=lambda_contrastive,
        pooling_method=pooling_method,
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        pos_weight=pos_weight
    )
    
    return model


# ============================================================================
# TESTING
# ============================================================================
def test_script():
    print("=" * 80)
    print("Testing CoevolutionBindingModel")
    print("=" * 80)
    
    # Create model
    model = create_model(
        encoder_name="facebook/esm2_t12_35M_UR50D",
        freeze_encoder=False,
        contrastive_dim=256,
        lambda_contrastive=0.33,
    )
    
    # Create dummy batch
    batch_size = 4
    seq_len = 256
    num_neg = 3
    
    batch = {
        'input_ids': torch.randint(0, 33, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'il6_start_idx': torch.tensor([150, 140, 145, 155]),
        'il6_end_idx': torch.tensor([200, 190, 195, 205]),
        'binding_label': torch.tensor([1, 0, 1, 0]),
        'has_negatives': torch.tensor([True, False, True, False]),
        'negative_input_ids': torch.randint(0, 33, (batch_size, num_neg, seq_len)),
        'negative_attention_mask': torch.ones(batch_size, num_neg, seq_len),
        'num_negatives': torch.tensor([3, 0, 2, 0]) 
    }
    
    # Set valid tokens
    batch['input_ids'][:, 0] = 0  # <cls>
    batch['input_ids'][:, -1] = 2  # <eos>
    batch['negative_input_ids'][:, :, 0] = 0
    batch['negative_input_ids'][:, :, -1] = 2
    
    # Forward pass
    print("\nRunning forward pass...")
    outputs = model(batch)
    
    print(f"\n{'='*80}")
    print("Outputs:")
    print("="*80)
    print(f"Total loss: {outputs['loss'].item():.4f}")
    print(f"  - Binding loss: {outputs['loss_binding']:.4f}")
    print(f"  - Contrastive loss: {outputs['loss_contrastive']:.4f}")
    print(f"\nBinding logits shape: {outputs['binding_logits'].shape}")
    print(f"Binding probs: {outputs['binding_probs']}")
    print(f"VHH embeddings shape: {outputs['z_vhh'].shape}")
    print(f"IL-6 embeddings shape: {outputs['z_il6'].shape}")
    
    # Test inference
    print(f"\n{'='*80}")
    print("Testing inference methods")
    print("="*80)
    
    probs = model.predict_binding(batch['input_ids'], batch['attention_mask'])
    print(f"Binding predictions shape: {probs.shape}")
    print(f"Binding predictions: {probs}")
    
  
    
    # Count parameters
    print(f"\n{'='*80}")
    print("Model Parameters")
    print("="*80)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = total_params - encoder_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Task heads parameters: {head_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
    
if __name__ == "__main__":
    test_script() 
    