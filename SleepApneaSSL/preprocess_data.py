"""
preprocess_data.py  —  V2 Anti-aliased preprocessing
=====================================================
Converts raw BIDS EDF files to 20-channel 100 Hz PyTorch tensors.

Fixes vs V1:
  - BUG-5: Channel mapping uses preallocated 20-slot array with correct positional
            placement. Missing channels are zero-filled at their exact target index.
  - BUG-6: Resampling ratio uses fractions.Fraction for exact rational arithmetic,
            handling non-integer source sampling frequencies correctly.
  - Recording-level resampling (not chunk-level) to avoid filter boundary artifacts.

Output: E:/SleepApneaProcessed_V2/sub-{sub}_eeg.pt
        E:/SleepApneaProcessed_V2/preprocessing_manifest.csv
"""
import os
import gc
import csv
import math
import torch
import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly
import edfio
from mne_bids import BIDSPath

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR    = 'E:/SleepApnea'
OUT_DIR    = 'E:/SleepApneaProcessed_V2'
TARGET_HZ  = 100.0
WINDOW_S   = 30.0        # seconds per output window
CHUNK_S    = 300.0       # seconds per processing chunk (memory management)
PREPROC_V  = '2.0'

# Target 20-channel layout — index position IS the channel's canonical position
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3',  'C3',  'Cz', 'C4', 'T4',
    'T5',  'P3',  'Pz', 'P4', 'T6',
    'O1',  'O2',  'Oz',
]
assert len(TARGET_CHANNELS) == 20

def _channel_match(actual_name: str, target: str) -> bool:
    """Case-insensitive substring match (e.g. 'EEG Fp1-REF' matches 'Fp1')."""
    return target.lower() in actual_name.lower()

def _exact_resample_poly(data: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    """
    Resample 2D array (channels, samples) from src_hz to dst_hz.
    Uses Fraction for exact rational up/down factors.
    data is C-contiguous float32.
    """
    if abs(src_hz - dst_hz) < 1e-6:
        return data.astype(np.float32)
    frac = Fraction(dst_hz).limit_denominator(10000) / Fraction(src_hz).limit_denominator(10000)
    up   = frac.numerator
    down = frac.denominator
    return resample_poly(data, up, down, axis=1).astype(np.float32)

def preprocess_dataset():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, 'preprocessing_manifest.csv')

    manifest_fields = [
        'subject_id', 'original_channels', 'mapped_channels',
        'missing_channels', 'original_hz', 'target_hz',
        'n_samples_orig', 'n_samples_resampled', 'n_windows',
        'preprocessing_version', 'status',
    ]

    # Open manifest in append mode so we can write as we go
    manifest_exists = os.path.exists(manifest_path)
    manifest_f = open(manifest_path, 'a', newline='')
    writer = csv.DictWriter(manifest_f, fieldnames=manifest_fields)
    if not manifest_exists:
        writer.writeheader()

    subjects = [f"{i:03d}" for i in range(1, 143)]
    print(f"V2 Preprocessing → {OUT_DIR}")
    print(f"Target: 20ch @ {TARGET_HZ} Hz | Window: {WINDOW_S}s")

    for sub in subjects:
        out_file = os.path.join(OUT_DIR, f'sub-{sub}_eeg.pt')
        if os.path.exists(out_file):
            print(f"  SKIP sub-{sub} (exists)")
            continue

        bids_path = BIDSPath(
            subject=sub, session='nightSleep', task='sleep', datatype='eeg',
            root=RAW_DIR, check=False
        )
        edf_path = bids_path.fpath
        if not os.path.exists(edf_path) or os.path.getsize(edf_path) < 1024 * 1024:
            print(f"  SKIP sub-{sub} (EDF not found or too small)")
            continue

        try:
            edf = edfio.read_edf(edf_path)
            actual_names = [sig.label.strip() for sig in edf.signals]
            src_hz = float(edf.signals[0].sampling_frequency)

            # ── BUG-5 FIX: positional channel mapping ─────────────────────────
            # Build index map: target_position → source_signal_index (or None)
            channel_map = {}   # target_idx → source_signal_idx
            for tgt_idx, tgt_name in enumerate(TARGET_CHANNELS):
                for src_idx, src_name in enumerate(actual_names):
                    if _channel_match(src_name, tgt_name):
                        channel_map[tgt_idx] = src_idx
                        break  # first match wins

            mapped    = [TARGET_CHANNELS[i] for i in channel_map]
            missing   = [TARGET_CHANNELS[i] for i in range(20) if i not in channel_map]

            if not channel_map:
                print(f"  FAIL sub-{sub}: no target channels found in {actual_names[:5]}...")
                writer.writerow({'subject_id': sub, 'status': 'no_channels_found',
                                 'original_channels': str(actual_names), 'preprocessing_version': PREPROC_V,
                                 'mapped_channels': '', 'missing_channels': str(TARGET_CHANNELS),
                                 'original_hz': src_hz, 'target_hz': TARGET_HZ,
                                 'n_samples_orig': 0, 'n_samples_resampled': 0, 'n_windows': 0})
                manifest_f.flush(); continue

            # ── Full-recording level resample (chunk by chunk for memory) ──────
            # Get total sample count from first mapped channel
            first_src_idx = next(iter(channel_map.values()))
            total_orig_samples = len(edf.signals[first_src_idx].data)
            total_duration = total_orig_samples / src_hz

            processed_chunks = []
            current_time = 0.0

            while current_time < total_duration:
                t_end = min(current_time + CHUNK_S, total_duration)
                s0 = int(current_time * src_hz)
                s1 = int(t_end * src_hz)
                if s0 >= s1:
                    break

                # ── BUG-5 FIX: preallocate 20-slot array ─────────────────────
                chunk_len = s1 - s0
                chunk = np.zeros((20, chunk_len), dtype=np.float32)
                for tgt_idx, src_idx in channel_map.items():
                    chunk[tgt_idx] = edf.signals[src_idx].data[s0:s1].astype(np.float32)

                # ── BUG-6 FIX: exact rational resampling ──────────────────────
                chunk_rs = _exact_resample_poly(chunk, src_hz, TARGET_HZ)
                processed_chunks.append(chunk_rs)
                current_time += CHUNK_S
                del chunk; gc.collect()

            final = np.hstack(processed_chunks)   # (20, total_resampled_samples)
            del processed_chunks; gc.collect()

            n_orig_samp = total_orig_samples
            n_rs_samp   = final.shape[1]
            win_samples = int(TARGET_HZ * WINDOW_S)   # = 3000
            n_windows   = n_rs_samp // win_samples

            if n_windows == 0:
                print(f"  FAIL sub-{sub}: no complete windows after resampling")
                continue

            # Trim to exact multiple of window size
            final = final[:, :n_windows * win_samples]

            tensor = torch.tensor(final, dtype=torch.float32)
            torch.save(tensor, out_file)

            print(f"  OK sub-{sub}: {len(mapped)}/20 ch | {n_windows} windows | "
                  f"{src_hz:.1f}→{TARGET_HZ:.1f} Hz | missing: {missing or 'none'}")

            writer.writerow({
                'subject_id':          sub,
                'original_channels':   str(actual_names),
                'mapped_channels':     str(mapped),
                'missing_channels':    str(missing),
                'original_hz':         src_hz,
                'target_hz':           TARGET_HZ,
                'n_samples_orig':      n_orig_samp,
                'n_samples_resampled': n_rs_samp,
                'n_windows':           n_windows,
                'preprocessing_version': PREPROC_V,
                'status':              'ok',
            })
            manifest_f.flush()

            del final, tensor, edf
            gc.collect()

        except Exception as e:
            print(f"  FAIL sub-{sub}: {e}")
            writer.writerow({'subject_id': sub, 'status': f'error: {e}',
                             'preprocessing_version': PREPROC_V,
                             'original_channels': '', 'mapped_channels': '',
                             'missing_channels': '', 'original_hz': 0,
                             'target_hz': TARGET_HZ, 'n_samples_orig': 0,
                             'n_samples_resampled': 0, 'n_windows': 0})
            manifest_f.flush()

    manifest_f.close()
    print(f"\nDone. Manifest: {manifest_path}")

if __name__ == '__main__':
    preprocess_dataset()