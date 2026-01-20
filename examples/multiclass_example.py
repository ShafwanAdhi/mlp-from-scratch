import numpy as np
from mlp_scratch import MLP

def main():
    # Create model
    model = MLP()
    model.sequential([4, 8, 'relu', 8, 'relu', 3, 'softmax'])
    model.set_loss('cce')
    model.set_optimizer('adam')
    
    # Dummy dataset (replace with real data)
    np.random.seed(42)
    X = np.random.randn(150, 4).tolist()
    
    # One-hot encoded labels
    y = []
    for i in range(150):
        if i < 50:
            y.append([1, 0, 0])
        elif i < 100:
            y.append([0, 1, 0])
        else:
            y.append([0, 0, 1])
    
    # Train
    print("Training multi-class classification...")
    model.fit(X, y, epoch=1000, learning_rate=0.01, lr_decay=0.995)
    
    # Test
    print("\nSample predictions:")
    for i in [0, 50, 100]:
        pred = model.forward(X[i])
        true_class = np.argmax(y[i])
        pred_class = np.argmax(pred)
        print(f"Sample {i}: True={true_class}, Predicted={pred_class}, Probs={pred}")

if __name__ == "__main__":
    main()
