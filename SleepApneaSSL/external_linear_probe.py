import os
import torch
import torch.nn as nn
import numpy as np
import mne
import warnings
import wfdb
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. SHHS Custom Dataset Loader ---
class SHHSDataset(Dataset):
    """
    Loads SHHS EDF files and maps ground-truth sleep stages directly from .hypn aux_note.
    """
    def __init__(self, record_base_paths, target_sfreq=100, window_sec=30):
        self.target_sfreq = target_sfreq
        self.window_len = target_sfreq * window_sec  # 100 Hz * 30s = 3000 samples
        
        # 11 SHHS channels -> 20-Channel Layout Map
        self.channel_map = {
            'EEG': 0, 'EEG(sec)': 1, 'EOG(L)': 2, 'EOG(R)': 3,
            'EMG': 4, 'ECG': 5, 'AIRFLOW': 6, 'THOR RES': 7,
            'ABDO RES': 8, 'SaO2': 9, 'PR': 10
        }
        
        self.samples = []
        self.labels = []
        self._load_and_process(record_base_paths)

    def _load_and_process(self, record_base_paths):
        for base_path in record_base_paths:
            edf_path = f"{base_path}.edf"
            hypn_path = f"{base_path}.hypn"
            
            if not os.path.exists(edf_path):
                print(f"[!] File not found: {edf_path}")
                continue

            # 1. Load Raw EDF
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            if raw.info['sfreq'] != self.target_sfreq:
                raw.resample(self.target_sfreq, verbose=False)

            data = raw.get_data()  # Shape: (11, Total_Samples)
            total_samples = data.shape[1]

            # 2. Map channels into 20-channel layout
            mapped_data = np.zeros((20, total_samples), dtype=np.float32)
            for i, ch in enumerate(raw.ch_names):
                if ch in self.channel_map:
                    mapped_data[self.channel_map[ch], :] = data[i, :]

            # 3. Extract Ground-Truth Annotations from aux_note
            stage_labels = []
            if os.path.exists(hypn_path):
                try:
                    ann = wfdb.rdann(base_path, 'hypn')
                    # aux_note contains strings like 'W', '1', '2', '3', '4', 'R'
                    # Binary Mapping: Wake ('W', '0') -> 0 | Sleep ('1', '2', '3', '4', 'R') -> 1
                    for note in ann.aux_note:
                        clean_note = note.strip()
                        if clean_note in ['W', '0']:
                            stage_labels.append(0)
                        elif clean_note == '1':
                            stage_labels.append(1)  # N1
                        elif clean_note == '2':
                            stage_labels.append(2)  # N2
                        elif clean_note in ['3', '4']:
                            stage_labels.append(3)  # N3 (Slow Wave Sleep)
                        elif clean_note == 'R':
                            stage_labels.append(4)  # REM
                        else:
                            stage_labels.append(0)  # Default fallback
                    print(f"[+] Loaded {len(stage_labels)} stage annotations from .hypn")
                except Exception as e:
                    print(f"[!] Error reading .hypn: {e}")

            # 4. Slice into 30-second windows and pair with ground-truth
            num_windows = total_samples // self.window_len
            for w in range(num_windows):
                start = w * self.window_len
                end = start + self.window_len
                window_tensor = mapped_data[:, start:end]

                if w < len(stage_labels):
                    label = stage_labels[w]
                    self.samples.append(window_tensor)
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
# --- 2. Feature Extraction & Linear Probe Logic ---
def extract_representations(encoder, dataloader, device):
    encoder.eval()
    latents, targets = [], []

    print("[*] Extracting frozen PhysioCLR representations from SHHS...")
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            with torch.amp.autocast('cuda'):
                z = encoder(x_batch)
            latents.append(z.cpu().numpy())
            targets.append(y_batch.numpy())

    return np.vstack(latents), np.concatenate(targets)


def fit_linear_probe(X_train, y_train, X_test, y_test):
    print("[*] Training Linear Probe on extracted embeddings...")
    clf = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    print("\n[+] Cross-Cohort (SHHS) Linear Probe Performance:")
    print(classification_report(y_test, preds, zero_division=0))

    try:
        present_classes = np.unique(y_test)
        if len(present_classes) > 2:
            # Binarize only against classes actually present in the test split
            y_test_bin = label_binarize(y_test, classes=clf.classes_)
            valid_cols = [i for i, c in enumerate(clf.classes_) if c in present_classes]
            
            if len(valid_cols) > 1 and probs.shape[1] == len(clf.classes_):
                auc_score = roc_auc_score(
                    y_test_bin[:, valid_cols], 
                    probs[:, valid_cols], 
                    multi_class='ovr', 
                    average='macro'
                )
                print(f"Macro AUROC Score: {auc_score:.4f}")
            else:
                print("Macro AUROC Score: N/A (Insufficient class variance in test split)")
        elif len(present_classes) == 2 and probs.shape[1] >= 2:
            print(f"AUROC Score: {roc_auc_score(y_test, probs[:, 1]):.4f}")
    except Exception as e:
        print(f"[!] AUROC calculation skipped: {e}")
# --- 3. Execution Pipeline ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Executing Cross-Cohort Evaluation on device: {device}")

    # Path to downloaded SHHS EDF
    import glob

    # Dynamically find all base record paths in the SHHS directory
    shhs_dir = r"E:\sleep-heart-health-study-psg-database-1.0.0"
    print(f"[*] Checking directory contents of: {shhs_dir}")
    print(f"[*] Top-level items: {os.listdir(shhs_dir)[:10]}")
    edf_files = glob.glob(os.path.join(shhs_dir, "**", "*.edf"), recursive=True)
    print(f"[*] Total EDF files found recursively: {len(edf_files)}")
    shhs_edf_paths = [os.path.splitext(f)[0] for f in edf_files]
    print(f"[*] Found {len(shhs_edf_paths)} SHHS records ready for batch evaluation.")

    print("[+] Loading pre-trained STFTEncoder2D model...")
    from model import STFTEncoder2D
    
    checkpoint_path = r'E:\SleepApnea\SleepApneaSSL\best_downstream_model.pth'
    encoder = STFTEncoder2D(in_channels=20).to(device)
    
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'classifier' not in k}
    encoder.load_state_dict(encoder_state, strict=False)

    print("[+] Processing SHHS EDF files...")
    dataset = SHHSDataset(shhs_edf_paths, target_sfreq=100, window_sec=30)
    print(f"[*] Total 30-second windows extracted: {len(dataset)}")

    if len(dataset) < 10:
        print("[!] Not enough windows generated. Check EDF path.")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Extract Features
    X, y = extract_representations(encoder, loader, device)

    # 80/20 Train-Test split for Linear Probe
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    fit_linear_probe(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()