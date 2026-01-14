import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass 
import random
try:
    from .tokenizer_esm2 import ESM2Tokenizer 
except ImportError:
    from tokenizer_esm2 import ESM2Tokenizer 

# ============================================================================
# DATASET CLASS
# ============================================================================
#Single training example. 
@dataclass
class BindingExample:                
    ab_id: str
    vhh_sequence: str
    ag_sequence: str
    ag_label: str
    mutation_position: Optional[int]
    binding_label: int
    hard_negatives: List[Dict]      # List of non-binding mutants 
    

class CoevolutionBindingDataset(Dataset): 
    def __init__(
        self,
        examples: List[BindingExample],
        tokenizer: ESM2Tokenizer,
        max_length: int = 512,
        max_negatives: int = 10 
    ):
        """
        Args:
            examples: List of training examples
            tokenizer: ESM2Tokenizer instance
            max_length: Maximum sequence length
            max_negatives: Maximum number of hard negatives per example
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_negatives = max_negatives 
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        ab_id = example.ab_id 
        ag_id = example.ag_label 
        # Tokenize VHH and antigen sequences
        vhh_tokens = self.tokenizer.encode(example.vhh_sequence, add_special_tokens=False)
        ag_tokens = self.tokenizer.encode(example.ag_sequence, add_special_tokens=False)
        
        # Build input: [CLS] VHH [EOS] AG [EOS]
        input_ids = [self.tokenizer.cls_token_id] + vhh_tokens + \
                    [self.tokenizer.eos_token_id] + ag_tokens + \
                    [self.tokenizer.eos_token_id]
        
        # Track IL-6 (antigen) sequence position
        il6_start_idx = 1 + len(vhh_tokens) + 1  # After [CLS] VHH [SEP]
        il6_end_idx = il6_start_idx + len(ag_tokens)
        
        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            il6_end_idx = min(il6_end_idx, self.max_length)
        
        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        
        # Prepare output dictionary
        output = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'il6_start_idx': torch.tensor(il6_start_idx, dtype=torch.long),
            'il6_end_idx': torch.tensor(il6_end_idx, dtype=torch.long),
            'binding_label': torch.tensor(example.binding_label, dtype=torch.long),
            'ab_id': ab_id,
            'ag_id': ag_id
        }
        
        # Add hard negatives for contrastive learning
        has_negatives = len(example.hard_negatives) > 0 and example.binding_label == 1
        output['has_negatives'] = torch.tensor(has_negatives, dtype=torch.bool)
        
        if has_negatives:
            negative_input_ids_list = []
            negative_attention_mask_list = []
            
            # Sample up to max_negatives
            # sampled_negatives = example.hard_negatives[:self.max_negatives]
            sampled_negatives = random.sample(
                example.hard_negatives,
                k=min(self.max_negatives, len(example.hard_negatives))
            )
            
            for neg in sampled_negatives:
                neg_ag_tokens = self.tokenizer.encode(neg['ag_sequence'], add_special_tokens=False)
                
                # Same VHH, different antigen
                neg_input_ids = [self.tokenizer.cls_token_id] + vhh_tokens + \
                            [self.tokenizer.eos_token_id] + neg_ag_tokens + \
                            [self.tokenizer.eos_token_id]
                
                # Truncate and pad
                if len(neg_input_ids) > self.max_length:
                    neg_input_ids = neg_input_ids[:self.max_length]
                
                neg_attention_mask = [1] * len(neg_input_ids)
                neg_padding_length = self.max_length - len(neg_input_ids)
                neg_input_ids += [self.tokenizer.pad_token_id] * neg_padding_length
                neg_attention_mask += [0] * neg_padding_length
                
                negative_input_ids_list.append(neg_input_ids)
                negative_attention_mask_list.append(neg_attention_mask)
            
            # Pad to max_negatives if needed
            while len(negative_input_ids_list) < self.max_negatives:
                negative_input_ids_list.append([self.tokenizer.pad_token_id] * self.max_length)
                negative_attention_mask_list.append([0] * self.max_length)
            
            output['negative_input_ids'] = torch.tensor(negative_input_ids_list, dtype=torch.long)
            output['negative_attention_mask'] = torch.tensor(negative_attention_mask_list, dtype=torch.float)
            output['num_negatives'] = torch.tensor(len(sampled_negatives), dtype=torch.long)
        else:
            # Dummy negatives (will be masked out)
            output['negative_input_ids'] = torch.zeros((self.max_negatives, self.max_length), dtype=torch.long)
            output['negative_attention_mask'] = torch.zeros((self.max_negatives, self.max_length), dtype=torch.float)
            output['num_negatives'] = torch.tensor(0, dtype=torch.long)
        
        return output



class RandomNegativeBindingDataset(Dataset): 
    def __init__(
        self,
        examples: List[BindingExample],
        tokenizer: ESM2Tokenizer,
        max_length: int = 512,
        max_negatives: int = 10 
    ):
        """
        Dataset with random negatives for ablation study.
        
        Args:
            examples: List of training examples
            tokenizer: ESM2Tokenizer instance
            max_length: Maximum sequence length
            max_negatives: Maximum number of random negatives per example
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_negatives = max_negatives
        
        # Collect all negative antigens from the dataset
        self.all_negative_ags = []
        for ex in examples:
            if ex.binding_label == 0:
                self.all_negative_ags.append({
                    'ag_sequence': ex.ag_sequence,
                    'ag_label': ex.ag_label
                })
        
        print(f"Collected {len(self.all_negative_ags)} negative antigens for random sampling")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        ab_id = example.ab_id 
        ag_id = example.ag_label 
        
        # Tokenize VHH and antigen sequences
        vhh_tokens = self.tokenizer.encode(example.vhh_sequence, add_special_tokens=False)
        ag_tokens = self.tokenizer.encode(example.ag_sequence, add_special_tokens=False)
        
        # Build input: [CLS] VHH [EOS] AG [EOS]
        input_ids = [self.tokenizer.cls_token_id] + vhh_tokens + \
                    [self.tokenizer.eos_token_id] + ag_tokens + \
                    [self.tokenizer.eos_token_id]
        
        # Track antigen sequence position
        il6_start_idx = 1 + len(vhh_tokens) + 1  # After [CLS] VHH [SEP]
        il6_end_idx = il6_start_idx + len(ag_tokens)
        
        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            il6_end_idx = min(il6_end_idx, self.max_length)
        
        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        
        # Prepare output dictionary
        output = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'il6_start_idx': torch.tensor(il6_start_idx, dtype=torch.long),
            'il6_end_idx': torch.tensor(il6_end_idx, dtype=torch.long),
            'binding_label': torch.tensor(example.binding_label, dtype=torch.long),
            'ab_id': ab_id,
            'ag_id': ag_id
        }
        
        # Add random negatives for positive examples only
        has_negatives = example.binding_label == 1 and len(self.all_negative_ags) > 0
        output['has_negatives'] = torch.tensor(has_negatives, dtype=torch.bool)
        
        if has_negatives:
            negative_input_ids_list = []
            negative_attention_mask_list = []
            
            # Sample random negatives from all negative antigens
            num_to_sample = min(self.max_negatives, len(self.all_negative_ags))
            sampled_negatives = random.sample(self.all_negative_ags, k=num_to_sample)
            
            for neg in sampled_negatives:
                neg_ag_tokens = self.tokenizer.encode(neg['ag_sequence'], add_special_tokens=False)
                
                # Same VHH, different antigen
                neg_input_ids = [self.tokenizer.cls_token_id] + vhh_tokens + \
                            [self.tokenizer.eos_token_id] + neg_ag_tokens + \
                            [self.tokenizer.eos_token_id]
                
                # Truncate and pad
                if len(neg_input_ids) > self.max_length:
                    neg_input_ids = neg_input_ids[:self.max_length]
                
                neg_attention_mask = [1] * len(neg_input_ids)
                neg_padding_length = self.max_length - len(neg_input_ids)
                neg_input_ids += [self.tokenizer.pad_token_id] * neg_padding_length
                neg_attention_mask += [0] * neg_padding_length
                
                negative_input_ids_list.append(neg_input_ids)
                negative_attention_mask_list.append(neg_attention_mask)
            
            # Pad to max_negatives if needed
            while len(negative_input_ids_list) < self.max_negatives:
                negative_input_ids_list.append([self.tokenizer.pad_token_id] * self.max_length)
                negative_attention_mask_list.append([0] * self.max_length)
            
            output['negative_input_ids'] = torch.tensor(negative_input_ids_list, dtype=torch.long)
            output['negative_attention_mask'] = torch.tensor(negative_attention_mask_list, dtype=torch.float)
            output['num_negatives'] = torch.tensor(len(sampled_negatives), dtype=torch.long)
        else:
            # Dummy negatives (will be masked out)
            output['negative_input_ids'] = torch.zeros((self.max_negatives, self.max_length), dtype=torch.long)
            output['negative_attention_mask'] = torch.zeros((self.max_negatives, self.max_length), dtype=torch.float)
            output['num_negatives'] = torch.tensor(0, dtype=torch.long)
        
        return output
    

# ============================================================================
# DATA PREPARATION FUNCTIONS  
# ============================================================================

def parse_mutation(ag_label: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Extract mutation info from labels like 'IL-6_D168A'"""
    if ag_label == 'IL-6_WTs':
        return None, None, None
    else:
        mutation_str = ag_label.split('_')[1]
        wt_aa = mutation_str[0]
        position = int(mutation_str[1:-1])
        mut_aa = mutation_str[-1]
        return position, wt_aa, mut_aa

def prepare_training_examples(df: pd.DataFrame) -> List[BindingExample]:
    """
    Prepare training examples with contrastive pairs.
    """
    # Parse mutation information
    df['mutation_position'] = df['Ag_label'].apply(lambda x: parse_mutation(x)[0])
    df['wt_aa'] = df['Ag_label'].apply(lambda x: parse_mutation(x)[1])
    df['mut_aa'] = df['Ag_label'].apply(lambda x: parse_mutation(x)[2])
    
    ab_groups = df.groupby('Ab_id')
    training_examples = []
    
    for ab_id, group in ab_groups:
        vhh_seq = group['VHH_sequence'].iloc[0]
        
        # Separate binders and non-binders
        binders = group[group['label'] == 1]
        non_binders = group[group['label'] == 0]
        
        # For each binding example, collect hard negatives
        for _, pos_row in binders.iterrows():
            hard_negatives = []
            for _, neg_row in non_binders.iterrows():
                # Ensure mutation_position is valid or None
                mut_pos = neg_row['mutation_position']
                if pd.notna(mut_pos):
                    try:
                        mut_pos = int(mut_pos)
                    except (ValueError, TypeError):
                        mut_pos = None
                
                hard_negatives.append({
                    'ag_sequence': neg_row['Ag_sequence'],
                    'ag_label': neg_row['Ag_label'],
                    'mutation_position': mut_pos
                })
            
            # Validate positive mutation position
            pos_mut_pos = pos_row['mutation_position']
            if pd.notna(pos_mut_pos):
                try:
                    pos_mut_pos = int(pos_mut_pos)
                except (ValueError, TypeError):
                    pos_mut_pos = None
            else:
                pos_mut_pos = None
            
            training_examples.append(BindingExample(
                ab_id=ab_id,
                vhh_sequence=vhh_seq,
                ag_sequence=pos_row['Ag_sequence'],
                ag_label=pos_row['Ag_label'],
                mutation_position=pos_mut_pos,
                binding_label=1,
                hard_negatives=hard_negatives
            ))
        
        # Also add negative examples (without hard negatives)
        for _, neg_row in non_binders.iterrows():
            # Validate negative mutation position
            neg_mut_pos = neg_row['mutation_position']
            if pd.notna(neg_mut_pos):
                try:
                    neg_mut_pos = int(neg_mut_pos)
                except (ValueError, TypeError):
                    neg_mut_pos = None
            else:
                neg_mut_pos = None
            
            training_examples.append(BindingExample(
                ab_id=ab_id,
                vhh_sequence=vhh_seq,
                ag_sequence=neg_row['Ag_sequence'],
                ag_label=neg_row['Ag_label'],
                mutation_position=neg_mut_pos,
                binding_label=0,
                hard_negatives=[]
            ))
    
    return training_examples

# ============================================================================
# DATALOADER CREATION
# ============================================================================

def create_dataloaders(
    train_pq_file: str,
    val_pq_file: str,
    test_pq_file: str,
    batch_size: int = 8,
    max_length: int = 512,
    max_negatives: int = 10,
    num_workers: int = 1,
    shuffle_train: bool = True,
    load_frac_data: float = 1.0,
    verbose: bool = False, 
    random_negatives: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders from CSV files.
    
    Args:
        train_pq_file: Path to training CSV
        val_pq_file: Path to validation CSV
        test_pq_file: Path to test CSV
        batch_size: Batch size for training
        max_length: Maximum sequence length
        max_negatives: Maximum number of hard negatives
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data
        load_frac_data: Load a fraction of dataset 
        random_negatives: use negatives from other antibodies 
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    tokenizer = ESM2Tokenizer()
    
    if verbose: print("Loading training data...")
    train_df = pd.read_parquet(train_pq_file)
    if load_frac_data<1.0:
        train_df = train_df.sample(frac=load_frac_data).reset_index(drop=True) 
    train_examples = prepare_training_examples(train_df)
    
    if random_negatives: 
        dataset_class = RandomNegativeBindingDataset
    else:
        dataset_class = CoevolutionBindingDataset
    
    train_dataset = dataset_class(
            examples=train_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            max_negatives=max_negatives
        )
    
    if verbose: print("Loading validation data...")
    val_df = pd.read_parquet(val_pq_file)
    if load_frac_data<1.0:
        val_df = val_df.sample(frac=load_frac_data).reset_index(drop=True) 
    val_examples = prepare_training_examples(val_df)
    val_dataset = dataset_class(
        examples=val_examples,
        tokenizer=tokenizer,
        max_length=max_length,
        max_negatives=max_negatives
    )
    
    if verbose: print("Loading test data...")
    test_df = pd.read_parquet(test_pq_file)
    if load_frac_data<1.0:
        test_df = test_df.sample(frac=load_frac_data).reset_index(drop=True) 
    test_examples = prepare_training_examples(test_df)
    test_dataset = dataset_class(
        examples=test_examples,
        tokenizer=tokenizer,
        max_length=max_length,
        max_negatives=max_negatives
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if verbose:
        print(f"Train examples: {len(train_dataset)}")
        print(f"Val examples: {len(val_dataset)}")
        print(f"Test examples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def create_single_dataloader(
    data_filepath_pq: str,
    batch_size: int = 16,
    max_length: int = 512,
    max_negatives: int = 10,
    num_workers: int = 1,
    shuffle: bool = False, 
    load_frac_data: float = 1.0,
    return_data: bool = False, 
    verbose: bool = False,
    random_negatives: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ Create a dataset and dataloader for a given dataset file. """
    tokenizer = ESM2Tokenizer() 
    if verbose: print("Loading data...")
    test_df = pd.read_parquet(data_filepath_pq)
    if load_frac_data<1.0:
        test_df = test_df.sample(frac=load_frac_data).reset_index(drop=True) 
    test_examples = prepare_training_examples(test_df)
    
    if random_negatives: 
        dataset_class = RandomNegativeBindingDataset
    else:
        dataset_class = CoevolutionBindingDataset
        
    test_dataset = dataset_class(
        examples=test_examples,
        tokenizer=tokenizer,
        max_length=max_length,
        max_negatives=max_negatives
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    if verbose:
        print(f"Total samples: {len(test_dataset)}")
    if return_data:
        return test_loader, test_dataset, test_examples
    return test_loader 

    

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_pq_file="../_data/ab_splits/train.parquet",
        val_pq_file="../_data/ab_splits/val.parquet",
        test_pq_file="../_data/ab_splits/test.parquet",
        batch_size=8,
        max_length=512,
        max_negatives=10,
        num_workers=1,
        load_frac_data=0.01,
        verbose=True
    )
    
    # Test a batch
    print("\nTesting data loading...")
    for batch in train_loader:
        print("\nBatch keys:", batch.keys()) 
        # ['input_ids', 'attention_mask', 'il6_start_idx', 'il6_end_idx', 'binding_label', 'ab_id', 'ag_id', 
        # 'has_negatives', 'negative_input_ids', 'negative_attention_mask', 'num_negatives']
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Attention mask shape:", batch['attention_mask'].shape)
        print("Binding labels shape:", batch['binding_label'].shape)
        print("Negative input IDs shape:", batch['negative_input_ids'].shape)
        print("Has negatives:", batch['has_negatives'].sum().item(), "out of", len(batch['has_negatives']))
        break
    
    # Test tokenizer
    print("\nTesting ESM-2 tokenizer...")
    tokenizer = ESM2Tokenizer()
    test_seq = "MKTAYIA"
    tokens = tokenizer.encode(test_seq)
    print(f"Sequence: {test_seq}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {tokenizer.decode(tokens)}")
    
    