import torch
import os
import matplotlib.pyplot as plt

def generate_heat_data(num_samples=1000, grid_size=64):
    """
    Generate synthetic data for the heat equation.
    Initial condition: a random hot spot (e.g. thermal runaway origin).
    Final condition: diffused heat map across the battery.
    """
    print(f"Generating {num_samples} samples of size {grid_size}x{grid_size}...")
    # Create coordinate grid
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    initial_states = torch.zeros(num_samples, grid_size, grid_size)
    final_states = torch.zeros(num_samples, grid_size, grid_size)
    
    for i in range(num_samples):
        # Random location for the thermal event
        cx = (torch.rand(1) - 0.5) * 1.5
        cy = (torch.rand(1) - 0.5) * 1.5
        
        # Initial hot spot (sharp, localized heat)
        initial_states[i] = torch.exp(-((X - cx)**2 + (Y - cy)**2) / 0.02)
        
        # Final state (diffused heat after some time t, spread out and slightly cooled down at center)
        final_states[i] = 0.5 * torch.exp(-((X - cx)**2 + (Y - cy)**2) / 0.2)
        
    return initial_states, final_states

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('../data', exist_ok=True)
    
    initial, final = generate_heat_data(1000, 64)
    
    # Save the dataset
    torch.save({'initial': initial, 'final': final}, '../data/synthetic_heat_data.pt')
    print("Data saved to data/synthetic_heat_data.pt")
    
    # Plot a sample to verify it looks correct
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(initial[0].numpy(), cmap='hot')
    plt.title("Initial State (t=0)")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(final[0].numpy(), cmap='hot')
    plt.title("Final State (t=1)")
    plt.colorbar()
    
    # Save the visualization
    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/sample_data.png')
    print("Sample visualization saved to results/sample_data.png")
