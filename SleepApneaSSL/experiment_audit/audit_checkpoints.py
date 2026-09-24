import os
import glob
import torch
import hashlib
from collections import defaultdict

RESULTS_DIR = r'E:\SleepApnea\SleepApneaSSL\results\multiseed'
SEEDS = [42, 123, 2025, 7, 13]
METHODS = ['A0_VanillaNTXent', 'A1_TemporalNTXent']

def get_checkpoint_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_parameter_norm(state_dict):
    norm = 0.0
    for name, param in state_dict.items():
        if 'weight' in name or 'bias' in name:
            norm += param.float().norm().item()
    return norm

def main():
    print("--- CHECKPOINT AUDIT ---")
    
    checkpoints = {}
    
    for seed in SEEDS:
        for method in METHODS:
            path = os.path.join(RESULTS_DIR, f"seed_{seed}", f"{method}_encoder.pth")
            if not os.path.exists(path):
                print(f"MISSING: {path}")
                continue
                
            ckpt_hash = get_checkpoint_hash(path)
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            norm = get_parameter_norm(state_dict)
            
            checkpoints[f"{method}_seed{seed}"] = {
                'path': path,
                'hash': ckpt_hash,
                'norm': norm
            }
            
            print(f"{method} Seed {seed}:")
            print(f"  Hash: {ckpt_hash}")
            print(f"  Norm: {norm:.4f}")
            
    # Check for duplicates
    hash_counts = defaultdict(list)
    for name, info in checkpoints.items():
        hash_counts[info['hash']].append(name)
        
    print("\n--- DUPLICATE CHECK ---")
    duplicates = False
    for h, names in hash_counts.items():
        if len(names) > 1:
            print(f"WARNING: DUPLICATE CHECKPOINTS FOUND!")
            print(f"Hash {h} is shared by: {names}")
            duplicates = True
            
    if not duplicates:
        print("All checkpoints are unique.")

if __name__ == '__main__':
    main()
