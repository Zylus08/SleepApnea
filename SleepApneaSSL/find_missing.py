import os

path = r"E:\SleepApnea"
all_subjects = [f"sub-{i:03d}" for i in range(1, 143)]
bad_subjects = []

for sub in all_subjects:
    edf_path = os.path.join(path, sub, "ses-nightSleep", "eeg", f"{sub}_ses-nightSleep_task-sleep_eeg.edf")
    
    if not os.path.exists(edf_path):
        bad_subjects.append(sub)
    else:
        size_mb = os.path.getsize(edf_path) / (1024 * 1024)
        if size_mb < 1.0:
            bad_subjects.append(sub)
            # Remove the corrupted LFS pointer so it downloads cleanly
            os.remove(edf_path)

print(f"Subjects needing re-download ({len(bad_subjects)} total):")
print(" ".join(bad_subjects))
print("\nCopy and paste this command into your terminal to fix them:")
print(f"openneuro-py download --dataset ds008108 --include {' '.join(bad_subjects)} --target_dir E:\\SleepApnea")