import numpy as np
from mlp_scratch import MLP

def main():
    # Create model
    model = MLP()
    model.sequential([2, 4, 'relu', 1, 'sigmoid'])
    model.set_loss('bce')
    model.set_optimizer('adam')
    
    # XOR dataset
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]
    
    # Train
    print("Training XOR problem...")
    model.fit(X, y, epoch=5000, learning_rate=0.01)
    
    # Test
    print("\nResults:")
    for x_test in X:
        pred = model.forward(x_test)
        print(f"Input: {x_test}, Prediction: {pred[0]:.4f}")

if __name__ == "__main__":
    main()
