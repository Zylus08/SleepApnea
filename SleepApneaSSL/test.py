import sys, os, glob, torch
import time

with open("test_log.txt", "w") as f:
    f.write("Starting...\n")
    f.flush()
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath('experiments/loss_ablation.py')))
        from experiments.loss_ablation import *
        f.write("Imported loss_ablation\n")
        f.flush()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        f.write(f"Device: {device}\n")
        f.flush()
        
        all_files = glob.glob(os.path.join(DATA_DIR, '*.pt'))[:1]
        f.write(f"Files: {all_files}\n")
        f.flush()
        
        ssl_files = all_files
        loss_fn = VanillaNTXentLoss(temperature=0.5)
        
        f.write("Starting pretrain_ssl...\n")
        f.flush()
        
        # We manually step through pretrain_ssl to see where it hangs
        set_seed(SEED)
        encoder = STFTEncoder2D(in_channels=20, embed_dim=128).to(device)
        model = SimCLR(encoder, projection_dim=64).to(device)
        masker = SpectralSubbandMasking(p=0.5).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scaler = GradScaler('cuda') if device.type == 'cuda' else None

        dataset = SSLStreamingDataset(ssl_files)
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, drop_last=True)
        
        f.write("Initialized models and loader\n")
        f.flush()
        
        epoch = 1
        model.train(); masker.train()
        ep_loss = ep_cont = ep_temp = 0.; n = 0
        
        f.write("Iterating loader...\n")
        f.flush()
        
        start_t = time.time()
        for x, sub_ids, time_idx, is_boundary in loader:
            f.write(f"Got batch in {time.time() - start_t:.2f}s. Shape: {x.shape}\n")
            f.flush()
            start_t = time.time()
            break
            
        f.write("Done test loop.\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
