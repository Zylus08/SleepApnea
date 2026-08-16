import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SleepApneaSSLDataset
from augmentations import EEGContrastiveTransform
from model import EEGEncoder, SimCLR

def info_nce_loss(proj1, proj2, temperature=0.1):
    # Normalize projections
    proj1 = F.normalize(proj1, dim=-1)
    proj2 = F.normalize(proj2, dim=-1)
    
    # Cosine similarity matrix
    batch_size = proj1.shape[0]
    out = torch.cat([proj1, proj2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    
    # Mask out self-similarity
    mask = (~torch.eye(2 * batch_size, device=sim_matrix.device, dtype=torch.bool)).float()
    sim_matrix = sim_matrix * mask
    
    # Calculate loss
    positives = torch.exp(torch.sum(proj1 * proj2, dim=-1) / temperature)
    positives = torch.cat([positives, positives], dim=0)
    
    loss = -torch.log(positives / sim_matrix.sum(dim=-1))
    return loss.mean()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Dataset and Dataloader
    transform = EEGContrastiveTransform()
    dataset = SleepApneaSSLDataset(
        processed_root='E:/SleepApneaProcessed', 
        subjects=[f"{i:03d}" for i in range(1, 143)], 
        transform=transform
    )
    
    # Update to 20 channels!
    encoder = EEGEncoder(in_channels=20)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # Initialize Model and Optimizer
    model = SimCLR(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Training Loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (view1, view2) in enumerate(dataloader):
            view1, view2 = view1.to(device), view2.to(device)
            
            optimizer.zero_grad()
            
            _, proj1 = model(view1)
            _, proj2 = model(view2)
            
            loss = info_nce_loss(proj1, proj2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.encoder.state_dict(), 'E:/SleepApnea/SleepApneaSSL/simclr_encoder.pth')
    print("Pre-training complete. Encoder weights saved.") 

if __name__ == '__main__':
    train()