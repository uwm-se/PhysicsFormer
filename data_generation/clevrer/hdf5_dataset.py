"""
Copyright (c) 2026 Style Machine LLC. All rights reserved.
Author: Jesse Pokora

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Style Machine LLC.
"""

"""
HDF5 Dataset for CLEVRER Training Data

Efficient loading of large CLEVRER training datasets stored in HDF5 format.
"""

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from typing import Dict, Optional, List


class CLEVRERHDF5Dataset(Dataset):
    """
    Dataset for loading CLEVRER training data from HDF5 format.
    
    The HDF5 file contains:
    - states: [N, seq_len, num_objects, state_dim] float32
    - masks: [N, seq_len, num_objects] float32
    - questions: [N] variable-length strings
    - answers: [N] variable-length strings
    - question_types: [N] variable-length strings
    - metadata: [N] JSON strings
    - numerical_targets: [N, 2] float32 (count, value)
    """
    
    def __init__(
        self,
        hdf5_path: str,
        max_samples: Optional[int] = None,
        question_types: Optional[List[str]] = None
    ):
        """
        Args:
            hdf5_path: Path to HDF5 file
            max_samples: Maximum number of samples to load (for debugging)
            question_types: Filter to specific question types
        """
        self.hdf5_path = hdf5_path
        self.hf = h5py.File(hdf5_path, 'r')
        
        self.num_samples = self.hf.attrs['num_samples']
        self.seq_len = self.hf.attrs['seq_len']
        self.num_objects = self.hf.attrs['num_objects']
        self.state_dim = self.hf.attrs['state_dim']
        
        # Build index (optionally filtered)
        if question_types is not None:
            self.indices = []
            for i in range(min(self.num_samples, max_samples or self.num_samples)):
                qt = self.hf['question_types'][i].decode('utf-8')
                if any(t in qt for t in question_types):
                    self.indices.append(i)
        else:
            self.indices = list(range(min(self.num_samples, max_samples or self.num_samples)))
        
        print(f"Loaded CLEVRER HDF5 dataset: {len(self.indices)} samples")
        print(f"  State shape: ({self.seq_len}, {self.num_objects}, {self.state_dim})")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict:
        real_idx = self.indices[idx]
        
        states = torch.from_numpy(self.hf['states'][real_idx].astype(np.float32))
        mask = torch.from_numpy(self.hf['masks'][real_idx].astype(np.float32))
        
        question = self.hf['questions'][real_idx]
        if isinstance(question, bytes):
            question = question.decode('utf-8')
        
        answer = self.hf['answers'][real_idx]
        if isinstance(answer, bytes):
            answer = answer.decode('utf-8')
        
        question_type = self.hf['question_types'][real_idx]
        if isinstance(question_type, bytes):
            question_type = question_type.decode('utf-8')
        
        metadata_str = self.hf['metadata'][real_idx]
        if isinstance(metadata_str, bytes):
            metadata_str = metadata_str.decode('utf-8')
        metadata = json.loads(metadata_str)
        
        numerical_targets = torch.from_numpy(
            self.hf['numerical_targets'][real_idx].astype(np.float32)
        )
        
        return {
            'states': states,
            'mask': mask,
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'metadata': metadata,
            'numerical_targets': {
                'count': numerical_targets[0].item(),
                'value': numerical_targets[1].item()
            }
        }
    
    def close(self):
        """Close the HDF5 file."""
        self.hf.close()
    
    def __del__(self):
        try:
            self.hf.close()
        except:
            pass


def collate_clevrer_hdf5(batch: List[Dict]) -> Dict:
    """Collate function for CLEVRER HDF5 dataset."""
    states = torch.stack([b['states'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])
    
    questions = [b['question'] for b in batch]
    answers = [b['answer'] for b in batch]
    question_types = [b['question_type'] for b in batch]
    metadata = [b['metadata'] for b in batch]
    
    numerical_targets = {
        'count': torch.tensor([b['numerical_targets']['count'] for b in batch]),
        'value': torch.tensor([b['numerical_targets']['value'] for b in batch])
    }
    
    return {
        'states': states,
        'masks': masks,
        'questions': questions,
        'answers': answers,
        'question_types': question_types,
        'metadata': metadata,
        'numerical_targets': numerical_targets
    }


if __name__ == '__main__':
    # Test the dataset
    dataset = CLEVRERHDF5Dataset('D:/physics-former-data/clevrer_training_expanded.h5', max_samples=100)
    
    print(f"\nDataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  States shape: {sample['states'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Question: {sample['question']}")
    print(f"  Answer: {sample['answer']}")
    print(f"  Type: {sample['question_type']}")
    
    dataset.close()
