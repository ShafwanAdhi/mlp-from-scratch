import numpy as np
from mlp_scratch import MLP

def main():
    # Create model
    model = MLP()
    model.sequential([1, 16, 'relu', 16, 'relu', 1, 'linear'])
    model.set_loss('mse')
    model.set_optimizer('adam')
    
    # Generate sine wave data
    X = [[x] for x in np.linspace(0, 2*np.pi, 100)]
    y = [[np.sin(x[0])] for x in X]
    
    # Train
    print("Training sine wave regression...")
    model.fit(X, y, epoch=2000, learning_rate=0.001)
    
    # Test
    print("\nSample predictions:")
    test_points = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    for x_test in test_points:
        pred = model.forward([x_test])
        actual = np.sin(x_test)
        print(f"x={x_test:.4f}, Predicted: {pred[0]:.4f}, Actual: {actual:.4f}")

if __name__ == "__main__":
    main()
