import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SleepApneaSSLDataset
from augmentations import EEGContrastiveTransform
from model import EEGEncoder, SimCLR

def info_nce_loss(proj1, proj2, temperature=0.1):
    proj1 = F.normalize(proj1, dim=-1)
    proj2 = F.normalize(proj2, dim=-1)
    
    batch_size = proj1.shape[0]
    out = torch.cat([proj1, proj2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    
    mask = (~torch.eye(2 * batch_size, device=sim_matrix.device, dtype=torch.bool)).float()
    sim_matrix = sim_matrix * mask
    
    positives = torch.exp(torch.sum(proj1 * proj2, dim=-1) / temperature)
    positives = torch.cat([positives, positives], dim=0)
    
    loss = -torch.log(positives / sim_matrix.sum(dim=-1))
    return loss.mean()

def train():
    print("[1/5] Initializing device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print(f"Using device: {device}")

    print("[2/5] Loading dataset & locking files...")
    transform = EEGContrastiveTransform()
    
    # Auto-grab the first 40 successfully processed subjects (~9.5 GB of RAM)
    import os
    all_processed = [f.split('_')[0].replace('sub-', '') for f in os.listdir('E:/SleepApneaProcessed') if f.endswith('.pt')]
    subset_subjects = all_processed[:40]
    
    dataset = SleepApneaSSLDataset(
        processed_root='E:/SleepApneaProcessed', 
        subjects=subset_subjects, 
        transform=transform
    )
    
    print("[3/5] Building DataLoader...")
    # Keep batch_size=8 for the 4GB GPU constraint
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
    print("[4/5] Initializing model on GPU...")
    encoder = EEGEncoder(in_channels=20)
    model = SimCLR(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    print("[5/5] Launching Training Loop!")
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                view1, view2 = batch
                view1, view2 = view1.to(device), view2.to(device)
                
                optimizer.zero_grad()
                
                _, proj1 = model(view1)
                _, proj2 = model(view2)
                
                loss = info_nce_loss(proj1, proj2)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            except Exception as e:
                print(f"\n❌ CRASH AT EPOCH {epoch+1}, BATCH {batch_idx}: {e}")
                import sys
                sys.exit(1)
                
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.encoder.state_dict(), 'E:/SleepApnea/SleepApneaSSL/simclr_encoder.pth')
    print("Pre-training complete. Encoder weights saved.") 

if __name__ == '__main__':
    train()