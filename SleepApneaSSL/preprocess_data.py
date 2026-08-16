import os
import torch
import numpy as np
import edfio
from mne_bids import BIDSPath
import gc

def preprocess_dataset_memory_safe():
    raw_dir = 'E:/SleepApnea'
    out_dir = 'E:/SleepApneaProcessed'
    os.makedirs(out_dir, exist_ok=True)

    target_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                       'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'Oz']

    subjects = [f"{i:03d}" for i in range(1, 143)]
    chunk_duration_sec = 300 
    target_sfreq = 100.0

    print(f"Starting native EDF stream preprocessing. Saving tensors to {out_dir}...")
    
    for sub in subjects:
        out_file = os.path.join(out_dir, f'sub-{sub}_eeg.pt')
        if os.path.exists(out_file):
            print(f"Skipping sub-{sub}: Already processed.")
            continue
            
        bids_path = BIDSPath(subject=sub, session='nightSleep', datatype='eeg', root=raw_dir, check=False)
        edf_path = bids_path.fpath
        
        if not os.path.exists(edf_path) or (os.path.getsize(edf_path) / (1024*1024)) < 1.0:
            continue
            
        try:
            # Native edfio read - extremely lightweight memory footprint
            edf = edfio.read_edf(edf_path)
            
            # Map channels
            actual_ch_names = [sig.label for sig in edf.signals]
            picks = []
            for target in target_channels:
                idx = next((i for i, ch in enumerate(actual_ch_names) if target.lower() in ch.lower()), None)
                if idx is not None:
                    picks.append(idx)
            
            if not picks:
                print(f"❌ Failed sub-{sub}: No target channels found.")
                continue
                
            # Assume all chosen EEG channels share the main sampling rate
            # (edfio stores physical data as float64 natively)
            sfreq = edf.signals[picks[0]].sampling_frequency
            total_duration = len(edf.signals[picks[0]].data) / sfreq
            downsample_factor = max(1, int(round(sfreq / target_sfreq)))
            
            processed_chunks = []
            current_time = 0.0
            
            while current_time < total_duration:
                start_idx = int(current_time * sfreq)
                stop_idx = int(min(current_time + chunk_duration_sec, total_duration) * sfreq)
                
                if start_idx >= stop_idx:
                    break
                    
                # Extract chunk directly from the edfio signal objects
                chunk_data = np.array([edf.signals[idx].data[start_idx:stop_idx] for idx in picks])
                
                # Numpy decimation
                if downsample_factor > 1:
                    chunk_data = chunk_data[:, ::downsample_factor]
                
                processed_chunks.append(chunk_data)
                current_time += chunk_duration_sec
                
            final_data = np.hstack(processed_chunks)
            
            # Enforce exactly 20 channels
            if final_data.shape[0] < 20:
                padding = np.zeros((20 - final_data.shape[0], final_data.shape[1]))
                final_data = np.vstack([final_data, padding])
            elif final_data.shape[0] > 20:
                final_data = final_data[:20, :]
                
            # Save
            tensor_data = torch.tensor(final_data, dtype=torch.float32)
            torch.save(tensor_data, out_file)
            
            print(f"✅ Processed and saved: sub-{sub}")
            
            del processed_chunks
            del final_data
            del edf
            gc.collect()
            
        except Exception as e:
            print(f"❌ Failed sub-{sub}: {e}")

if __name__ == '__main__':
    preprocess_dataset_memory_safe()