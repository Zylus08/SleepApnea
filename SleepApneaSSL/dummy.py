import mne
import numpy as np
from mne_bids import BIDSPath, write_raw_bids

# Create fake EEG data: 32 channels, 100Hz, 30 minutes long
sfreq = 100
n_channels = 32
n_samples = sfreq * 60 * 30 
ch_names = [f'EEG {i:03}' for i in range(n_channels)]

for sub_id in ['001', '002']:
    # Generate random noise mimicking EEG
    data = np.random.randn(n_channels, n_samples) * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    # Save to BIDS format
    bids_path = BIDSPath(subject=sub_id, session='nightSleep', task='sleep',
                         datatype='eeg', root='E:/SleepApneaDummy')
    
    # FIX: Added allow_preload=True
    write_raw_bids(raw, bids_path, format='EDF', allow_preload=True, overwrite=True)

print("Dummy dataset created at E:/SleepApneaDummy")