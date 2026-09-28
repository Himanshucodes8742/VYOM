import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from registration_engine.learned_descriptor import train_descriptor

if __name__ == "__main__":
    data_path = os.path.join("data", "training", "patch_triplets.npz")
    save_path = os.path.join("registration_engine", "models", "patch_descriptor.pt")
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find training data at {data_path}")
        sys.exit(1)
        
    print(f"Starting descriptor training using triplets from {data_path}...")
    train_descriptor(data_path=data_path, save_path=save_path, epochs=30, batch_size=64)
