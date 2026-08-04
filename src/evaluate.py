import torch
import matplotlib.pyplot as plt
from fno_model import FNO2d
import os

def evaluate_model():
    print("Loading test data and model...")
    data_path = '../data/synthetic_heat_data.pt'
    model_path = '../models/fno_battery_model.pt'
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Missing data or model. Run data_generation.py and train.py first!")
        return
        
    data = torch.load(data_path)
    test_x, test_y = data['initial'][800:810], data['final'][800:810] # grab 10 samples
    
    model = FNO2d(modes=12, width=32)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        predictions = model(test_x)
        
    # Plot results
    print("Generating evaluation plots...")
    os.makedirs('../results', exist_ok=True)
    
    for i in range(3): # Plot first 3 examples
        plt.figure(figsize=(15, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(test_x[i].numpy(), cmap='hot')
        plt.title("Initial State (t=0)")
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(test_y[i].numpy(), cmap='hot')
        plt.title("True Final State (FEA)")
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i].numpy(), cmap='hot')
        plt.title("FNO Prediction")
        plt.colorbar()
        
        plt.savefig(f'../results/evaluation_sample_{i+1}.png')
        plt.close()
        
    print("Evaluation complete! Check the results/ folder for images.")

if __name__ == "__main__":
    evaluate_model()
