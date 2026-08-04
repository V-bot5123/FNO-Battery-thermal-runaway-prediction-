import torch
import torch.nn as nn
import torch.optim as optim
from fno_model import FNO2d
import os

def train_model():
    print("Loading data...")
    data_path = '../data/synthetic_heat_data.pt'
    if not os.path.exists(data_path):
        print("Data not found. Run data_generation.py first!")
        return
        
    data = torch.load(data_path)
    # Use first 800 for training, 200 for testing
    train_x, train_y = data['initial'][:800], data['final'][:800]
    
    # Initialize Model, Loss function, and Optimizer
    model = FNO2d(modes=12, width=32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20 # Reduced for faster demo!
    batch_size = 32
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Simple batching
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / (len(train_x) / batch_size)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
            
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/fno_battery_model.pt')
    print("Training complete! Model saved to models/fno_battery_model.pt")

if __name__ == "__main__":
    train_model()
