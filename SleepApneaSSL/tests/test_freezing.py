import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import STFTEncoder2D
from downstream_finetune import SleepApneaClassifier

def test_freezing():
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    model = SleepApneaClassifier(encoder, finetune_mode='frozen')
    
    # Save initial state
    initial_state = {name: param.clone() for name, param in model.encoder.named_parameters()}
    initial_buffers = {name: buf.clone() for name, buf in model.encoder.named_buffers()}
    
    # Simulate a training step
    model.train()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)
    
    dummy_input = torch.randn(4, 20, 3000)
    dummy_target = torch.empty(4).random_(2)
    
    optimizer.zero_grad()
    logits = model(dummy_input)
    loss = nn.BCEWithLogitsLoss()(logits, dummy_target)
    loss.backward()
    optimizer.step()
    
    # Check gradients
    for name, param in model.encoder.named_parameters():
        assert param.grad is None, f"Encoder parameter {name} has a gradient!"
        
    # Check parameter equality
    for name, param in model.encoder.named_parameters():
        assert torch.equal(param, initial_state[name]), f"Encoder parameter {name} changed!"
        
    # Check BN buffer equality
    for name, buf in model.encoder.named_buffers():
        assert torch.equal(buf, initial_buffers[name]), f"Encoder buffer {name} changed!"
        
    print("Freezing test PASSED.")

if __name__ == "__main__":
    test_freezing()
