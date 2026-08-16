import os
import glob

def verify_dataset(path):
    print(f"Scanning directory: {path}...\n")
    
    if not os.path.exists(path):
        print(f"ERROR: Path {path} does not exist.")
        return

    # 1. Check Subject Count
    subjects = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("sub-")]
    subject_count = len(subjects)
    print(f"Total 'sub-*' folders found: {subject_count} / 142")
    
    if subject_count < 142:
        print(f"➔ WARNING: You are missing {142 - subject_count} subjects.")
    elif subject_count == 142:
        print("➔ SUCCESS: All 142 subjects are present.")

    # 2. Check for EDF files and their sizes
    edf_files = glob.glob(os.path.join(path, "sub-*", "ses-nightSleep", "eeg", "*.edf"))
    print(f"\nTotal '.edf' files found: {len(edf_files)} / {subject_count}")
    
    missing_edf = subject_count - len(edf_files)
    if missing_edf > 0:
        print(f"➔ WARNING: {missing_edf} subjects are missing their nocturnal EEG files.")

    # 3. Check for Git LFS Pointers (File size check)
    if edf_files:
        lfs_pointers = 0
        valid_files = 0
        total_size_gb = 0
        
        for edf in edf_files:
            size_mb = os.path.getsize(edf) / (1024 * 1024)
            total_size_gb += size_mb / 1024
            if size_mb < 1.0:
                lfs_pointers += 1
            else:
                valid_files += 1
                
        print("\nFile Size Verification:")
        if lfs_pointers > 0:
            print(f"➔ CRITICAL FAILURE: {lfs_pointers} files are extremely small (< 1MB). These are Git LFS pointers, not actual data. You need to re-download these.")
        if valid_files > 0:
            print(f"➔ SUCCESS: {valid_files} files are legitimate EEG data.")
            print(f"➔ Total downloaded EEG data size: {total_size_gb:.2f} GB")
    else:
        print("\n➔ CRITICAL FAILURE: No EDF files were found to verify.")

if __name__ == "__main__":
    dataset_path = r"E:\SleepApnea"
    verify_dataset(dataset_path)