# ============================================================================
# ESM-2 TOKENIZER (Exact vocabulary and IDs)
# ============================================================================


from typing import List 


class ESM2Tokenizer:
    """
    ESM-2 tokenizer with exact vocabulary matching the official implementation.
    Vocabulary size: 33 tokens
    """
    def __init__(self):
        # ESM-2 standard vocabulary (indices 0-32)
        self.vocab = {
            '<cls>': 0,
            '<pad>': 1,
            '<eos>': 2,
            '<unk>': 3,
            'L': 4,
            'A': 5,
            'G': 6,
            'V': 7,
            'S': 8,
            'E': 9,
            'R': 10,
            'T': 11,
            'I': 12,
            'D': 13,
            'P': 14,
            'K': 15,
            'Q': 16,
            'N': 17,
            'F': 18,
            'Y': 19,
            'M': 20,
            'H': 21,
            'W': 22,
            'C': 23,
            'X': 24,  # Unknown amino acid
            'B': 25,  # Aspartic acid or Asparagine
            'U': 26,  # Selenocysteine
            'Z': 27,  # Glutamic acid or Glutamine
            'O': 28,  # Pyrrolysine
            '.': 29,  # Gap
            '-': 30,  # Gap
            '<null_1>': 31,
            '<mask>': 32,
        }
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Special tokens
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.mask_token_id = 32
        
    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a protein sequence to token IDs.
        
        Args:
            sequence: Protein sequence string (e.g., "MKTAYIA")
            add_special_tokens: Whether to add <cls> and <eos>
            
        Returns:
            List of token IDs
        """
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.cls_token_id)
        
        for aa in sequence:
            tokens.append(self.vocab.get(aa.upper(), self.unk_token_id))
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to sequence."""
        sequence = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '<unk>')
            if token not in ['<cls>', '<pad>', '<eos>', '<unk>']:
                sequence.append(token)
        return ''.join(sequence)
    
    def __len__(self):
        return len(self.vocab)
    
