import os
import torch
import shutil

def audit_dataset(processed_dir, min_windows=480, samples_per_window=3000):
    corrupt_dir = os.path.join(processed_dir, 'corrupted_files')
    os.makedirs(corrupt_dir, exist_ok=True)
    
    all_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
    purged = 0
    
    print("--- AUDITING DATASET INTEGRITY ---")
    for file in all_files:
        file_path = os.path.join(processed_dir, file)
        try:
            data = torch.load(file_path, weights_only=True)
            total_samples = data.shape[1]
            total_windows = total_samples // samples_per_window
            
            if total_windows < min_windows:
                print(f"[PURGE] {file} has only {total_windows} windows. Moving to corrupted folder.")
                shutil.move(file_path, os.path.join(corrupt_dir, file))
                purged += 1
        except Exception as e:
            print(f"[ERROR] Could not read {file}: {e}")
            
    print(f"\nAudit complete. Purged {purged} truncated patients.")

if __name__ == '__main__':
    audit_dataset('E:/SleepApneaProcessed')