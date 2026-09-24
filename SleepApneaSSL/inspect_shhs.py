"""Inspect SHHS record 0000 to understand available annotations and channels."""
import wfdb
import mne
import os

base = 'E:/sleep-heart-health-study-psg-database-1.0.0/0000'
edf_path = base + '.edf'

# 1. Check EDF channels
print("=== EDF Channels ===")
raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
print(f"Channels ({len(raw.ch_names)}):", raw.ch_names)
print(f"Sample rate: {raw.info['sfreq']}")
print(f"Duration: {raw.n_times / raw.info['sfreq']:.0f} seconds")

# 2. Check respiratory annotations
print("\n=== Respiratory Annotations (.resp) ===")
try:
    ann = wfdb.rdann(base, 'resp')
    print(f"Total events: {len(ann.aux_note)}")
    unique = set([n.strip() for n in ann.aux_note])
    print(f"Unique labels: {unique}")
    print(f"First 20 annotations: {ann.aux_note[:20]}")
    print(f"First 20 sample indices: {ann.sample[:20]}")
except Exception as e:
    print(f"Error: {e}")

# 3. Check hypnogram annotations
print("\n=== Hypnogram Annotations (.hypn) ===")
try:
    hypn = wfdb.rdann(base, 'hypn')
    print(f"Total stages: {len(hypn.aux_note)}")
    unique_stages = set([n.strip() for n in hypn.aux_note])
    print(f"Unique stages: {unique_stages}")
    print(f"First 20: {hypn.aux_note[:20]}")
except Exception as e:
    print(f"Error: {e}")

# 4. Check arousal annotations
print("\n=== Arousal Annotations (.arou) ===")
try:
    arou = wfdb.rdann(base, 'arou')
    print(f"Total events: {len(arou.aux_note)}")
    unique_arou = set([n.strip() for n in arou.aux_note])
    print(f"Unique labels: {unique_arou}")
except Exception as e:
    print(f"Error: {e}")
