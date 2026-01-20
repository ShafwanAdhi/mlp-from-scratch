# MLP From Scratch

> A production-grade Multi-Layer Perceptron implementation built from first principles using pure NumPy, demonstrating deep understanding of neural network internals and ML engineering fundamentals.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-orange)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Motivation](#-motivation)
- [Installation](#-installation)
- [Core Features](#-core-features)
- [Technical Deep Dive](#-technical-deep-dive)
- [Quick Start](#-quick-start)
- [Performance With PyTorch Comparison](#-Performance-With-PyTorch-Comparison)
- [Design Trade-offs](#-design-trade-offs)

---

## Project Overview

MLP-from-scratch is a fully-featured neural network library implemented entirely in NumPy, designed to bridge the gap between theoretical understanding and practical implementation of deep learning systems. This project demonstrates end-to-end ML engineering capabilities, from mathematical foundations to production-ready code.

**Technical Value:**
- **Educational Transparency**: Every operation is explicit and traceable, unlike black-box frameworks
- **Performance Competitive**: Achieves 4.5x faster training than PyTorch on small-scale problems
- **Production Patterns**: Implements industry-standard practices (modular design, weight initialization strategies, optimizer variants)
- **Minimal Dependencies**: Only requires NumPy, making it lightweight and easily deployable

This implementation proves that understanding the fundamentals enables building performant systems without relying on abstraction layers.

---

## Motivation

### Why Build from Scratch?

Modern ML frameworks like PyTorch and TensorFlow are powerful but abstract away critical implementation details. This project was built to:

1. Master the Fundamentals: Implementing backpropagation, gradient descent, and optimization algorithms from scratch solidifies understanding that's often lost when using high-level APIs.
2. Performance Understanding: By controlling every computation, I can analyze exactly where time is spent and make informed optimization decisions.
3. Real-world Problem Solving: Demonstrates ability to build production-grade systems without dependency on third-party frameworks—valuable in resource-constrained or specialized environments.

### Why Not Use Frameworks?

- Learning Depth: Frameworks hide complexities that ML engineers must understand
- Performance Control: Direct NumPy operations can be faster for specific use cases
- Customization: Full control over every aspect of training and inference
- Interview Relevance: Many technical interviews assess understanding of fundamentals, not framework APIs

---

## Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/ShafwanAdhi/mlp-from-scratch.git
```

### Local Development

```bash
git clone https://github.com/ShafwanAdhi/mlp-from-scratch.git
cd mlp-from-scratch
pip install -e .
```

### Dependencies

- Python >= 3.7
- NumPy >= 1.19.0

That's it! No CUDA, no TensorFlow, no PyTorch.

---

## Core Features

### Neural Network Components

- **Activation Functions**
  - ReLU, Leaky ReLU (α=0.01)
  - Tanh, Sigmoid
  - Softmax (with numerical stability)
  - Linear (identity)

- **Loss Functions**
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Binary Cross-Entropy (BCE)
  - Categorical Cross-Entropy (CCE)

- **Optimization Algorithms**
  - Stochastic Gradient Descent (SGD)
  - SGD with Momentum (β=0.9)
  - RMSprop (β=0.9)
  - Adam (β1=0.9, β2=0.999)

- **Weight Initialization**
  - Xavier/Glorot initialization (for sigmoid/tanh)
  - He initialization (for ReLU variants)

### Engineering Features

- **Automatic Gradient Computation**: Full backpropagation implementation
- **Flexible Architecture**: Define any network topology via simple list syntax
- **Learning Rate Scheduling**: Exponential decay support
- **Numerical Stability**: Clipping and epsilon terms prevent overflow/underflow
- **Validation Coupling**: Automatic pairing of activation-loss combinations (e.g., sigmoid-BCE)

---

## Technical Deep Dive

### Backpropagation: Conceptual Explanation

Backpropagation is the core algorithm that enables neural networks to learn. Here's how it works without heavy math:

**The Core Idea:**
Think of a neural network as a chain of functions. When we make a prediction, data flows forward through this chain. When we want to improve the network, we need to know how each weight contributed to the error. Backpropagation calculates this by working backwards through the chain.

**Step-by-Step Process:**

1. **Forward Pass**: 
   - Data flows through layers: Input → Hidden Layers → Output
   - Each layer computes: `activation(weights × input + bias)`
   - We save all intermediate values (crucial for backwards pass)

2. **Error Calculation**:
   - Compare prediction with actual answer
   - Measure "how wrong" we were using a loss function

3. **Backward Pass** (the "learning" part):
   - Output Layer: Calculate how much each neuron contributed to the error
   - Hidden Layers: Work backwards, asking "how much did this neuron's output affect the final error?"
   - Chain Rule: Each layer's gradient depends on the layer after it (hence "chain" rule)

4. **Gradient Computation**:
   - For each weight: "If I slightly change this weight, how much does the error change?"
   - This sensitivity is the gradient

**Implementation Strategy:**
```python
# Simplified conceptual flow
def backprop(self):
    # Start from output layer error
    gradient = loss_derivative(prediction, true_value)
    
    # Work backwards through layers
    for layer in reversed(layers):
        # How much did this layer contribute?
        gradient = gradient * activation_derivative(layer)
        
        # Calculate weight updates
        weight_gradient = gradient ⊗ previous_activation
        
        # Pass gradient to previous layer
        gradient = weights.T @ gradient
```

### Optimizer Algorithms: Intuitive Explanation

Optimizers determine **how** we update weights using the gradients from backpropagation.

#### 1. **SGD (Stochastic Gradient Descent)**
*The baseline approach.*

- Intuition: Walk downhill in the direction that reduces error
- Update Rule: `weight = weight - learning_rate × gradient`
- Analogy: Walking straight downhill on a mountain
- Limitation: Can oscillate or get stuck in local minima

#### 2. **SGD with Momentum**
*Adding inertia to gradient descent.*

- Intuition: Remember previous directions and keep some momentum
- Benefit: Smooths out oscillations, accelerates in consistent directions
- Analogy: A ball rolling down a hill (builds up speed)
- Implementation: 
  ```python
  velocity = β × velocity + learning_rate × gradient
  weight = weight - velocity
  ```

#### 3. **RMSprop**
*Adaptive learning rates per parameter.*

- Intuition: Give smaller updates to weights that change frequently, larger updates to stable ones
- Benefit: Different learning rates for different parameters
- Analogy: Adjusting step size based on terrain roughness
- Key Mechanism: Tracks exponential average of squared gradients

#### 4. **Adam (Adaptive Moment Estimation)**
*Best of both worlds: momentum + adaptive learning rates.*

- *ntuition: Combines momentum (direction memory) with adaptive rates (per-parameter scaling)
- Why It's Popular: Robust across different problems, rarely needs tuning
- Components:
  - First moment: Average of gradients (momentum)
  - Second moment: Average of squared gradients (adaptive rates)
- Bias Correction: Adjusts for initialization bias in early training

**Performance Comparison in Practice:**
- **SGD**: Slow but steady, good for simple problems
- **Momentum**: Faster convergence, better for problems with ravines
- **RMSprop**: Great for RNNs and non-stationary problems
- **Adam**: Default choice for most deep learning tasks

---

## Quick Start

### Training Example: XOR Problem

```python
import numpy as np
from mlp_scratch import MLP

# Initialize model
model = MLP()

# Define architecture: [input_dim, hidden_dim, activation, ..., output_dim, activation]
model.sequential([2, 4, 'relu', 1, 'sigmoid'])

# Configure training
model.set_loss('bce')
model.set_optimizer('adam')

# Prepare XOR dataset
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [[0], [1], [1], [0]]

# Train
model.fit(X_train, y_train, epoch=5000, learning_rate=0.01)

# Inference
for x in X_train:
    prediction = model.forward(x)
    print(f"Input: {x} → Prediction: {prediction[0]:.4f}")
```

**Output:**
```
Input: [0, 0] → Prediction: 0.0021
Input: [0, 1] → Prediction: 0.9978
Input: [1, 0] → Prediction: 0.9981
Input: [1, 1] → Prediction: 0.0019
```

### Multi-class Classification

```python
# For 3-class classification problem
model = MLP()
model.sequential([4, 16, 'relu', 8, 'relu', 3, 'softmax'])
model.set_loss('cce')  # Categorical cross-entropy
model.set_optimizer('adam')

# One-hot encoded labels
X_train = [
    #class1
    [1.0, 0.2, 0.1, 0.0],   
    [0.9, 0.1, 0.2, 0.1],  
    #class1
    [0.1, 1.0, 0.8, 0.9],   
    [0.2, 0.9, 0.7, 0.8], 
    #class2
    [0.0, 0.1, 1.0, 0.9],   
    [0.1, 0.2, 0.9, 1.0]   
]
y_train = [
    #class0
    [1, 0, 0],  
    [1, 0, 0],
    #class1
    [0, 1, 0],  
    [0, 1, 0],
    #class2
    [0, 0, 1],  
    [0, 0, 1]
]
model.fit(X_train, y_train, epoch=2000, learning_rate=0.01, lr_decay=0.995)
```

### Regression Task

```python
# Predict continuous values
model = MLP()
model.sequential([1, 16, 'relu', 16, 'relu', 1, 'linear'])
model.set_loss('mse')
model.set_optimizer('adam')

# Sine wave approximation
X = [[x] for x in np.linspace(0, 2*np.pi, 100)]
y = [[np.sin(x[0])] for x in X]

model.fit(X, y, epoch=2000, learning_rate=0.001)
```

---

## Performance With PyTorch Comparison

### Benchmark Setup

**Dataset:** sklearn `make_circles` (1000 samples, noise=0.1, factor=0.5)
- Training set: 800 samples
- Test set: 200 samples

**Architecture:** [2, 8, 'relu', 8, 'relu', 1, 'sigmoid']

**Training Configuration:**
- Epochs: 1000
- Learning Rate: 0.01
- LR Decay: 0.999
- Optimizer: Adam

### Results Comparison

| Metric                         | Custom MLP | PyTorch MLP | Winner        |
|--------------------------------|-----------|-------------|---------------|
| **Training Performance**       |           |             |               |
| Final Loss                     | 0.000002  | 0.000000    | PyTorch ✓     |
| Total Training Time (s)        | **185.25**| 827.77      | **Custom ✓**  |
| Avg Time per Epoch (s)         | **0.1852**| 0.8278      | **Custom ✓**  |
| **Speed Improvement**          | **4.47x faster** | —    | **Custom ✓**  |
|                                |           |             |               |
| **Accuracy Metrics**           |           |             |               |
| Train Accuracy                 | 100.00%   | 100.00%     | Tie           |
| Test Accuracy                  | 97.00%    | **98.50%**  | PyTorch ✓     |
| Train Error Rate               | 0.00%     | 0.00%       | Tie           |
| Test Error Rate                | 3.00%     | **1.50%**   | PyTorch ✓     |
|                                |           |             |               |
| **Classification Metrics**     |           |             |               |
| Precision                      | 97.96%    | **100.00%** | PyTorch ✓     |
| Recall                         | 96.00%    | **97.00%**  | PyTorch ✓     |
| F1-Score                       | 96.97%    | **98.48%**  | PyTorch ✓     |
|                                |           |             |               |
| **Inference Performance**      |           |             |               |
| Latency (ms/sample)            | **0.0259**| 0.0741      | **Custom ✓**  |
| **Speed Improvement**          | **2.86x faster** | —    | **Custom ✓**  |

<img width="500" height="370" alt="loss_curve" src="https://github.com/user-attachments/assets/cea190bd-757c-440c-a51a-21900e4419a4" />


### Key Insights

#### Custom MLP Advantages

1. **Training Speed (4.47x faster)**
   - **Why**: Direct NumPy operations without framework overhead
   - **Impact**: Faster iteration during experimentation and hyperparameter tuning
   - **Use Case**: Ideal for rapid prototyping and small-medium datasets

2. **Inference Speed (2.86x faster)**
   - **Why**: Minimal abstraction layers, no computational graph construction
   - **Impact**: Lower latency in production serving
   - **Use Case**: Edge deployment, real-time systems with strict latency requirements

3. **Memory Efficiency**
   - No GPU memory allocation overhead
   - Smaller binary footprint (NumPy vs PyTorch dependencies)
   - Predictable memory usage patterns

#### Custom MLP Limitations

1. **Slightly Lower Accuracy (97% vs 98.5%)**
   - **Gap**: 1.5 percentage points on test set
   - **Cause**: Potentially different numerical precision or initialization
   - **Acceptable Trade-off**: For many applications, 97% accuracy with 4x speed is preferable

2. **Generalization**
   - PyTorch's superior test accuracy suggests better generalization
   - Possible improvements: Add dropout, batch normalization, better regularization

### When to Use Custom MLP vs PyTorch

| Use Custom MLP When:                | Use PyTorch When:                   |
|-------------------------------------|-------------------------------------|
| Dataset fits in memory (< 10K samples) | Large-scale datasets (100K+ samples) |
| Latency is critical (< 1ms)         | Accuracy is paramount               |
| No GPU available                    | GPU acceleration needed             |
| Educational/interview purposes      | Production deep learning systems    |
| Embedded systems deployment         | Complex architectures (CNNs, Transformers) |

### Objective Comparison

**Custom MLP wins on:**
- Training speed (small datasets)
- Inference latency
- Simplicity and transparency
- Minimal dependencies

**PyTorch wins on:**
- Final model accuracy
- Generalization capability
- GPU acceleration
- Advanced features (distributed training, automatic differentiation)

---

## Design Trade-offs

### Conscious Limitations

As an ML engineer, I'm aware of the following trade-offs made in this implementation:

#### 1. **Full-Batch Training Only**
- **Current**: Processes entire dataset per epoch
- **Limitation**: Memory-intensive for large datasets
- **Trade-off**: Simplicity vs scalability
- **Future Fix**: Implement mini-batch training with data loaders

#### 2. **CPU-Only Execution**
- **Current**: Pure NumPy (CPU-bound)
- **Limitation**: No GPU acceleration
- **Trade-off**: Portability vs speed for large models
- **Context**: Acceptable for small-medium networks; PyTorch/JAX needed for deep models

#### 3. **No Regularization Techniques**
- **Missing**: Dropout, L1/L2 regularization, batch normalization
- **Impact**: Potential overfitting on complex datasets
- **Explanation**: Focus on core algorithms first; regularization is additive
- **Roadmap**: Next major feature addition

#### 4. **Sequential Architecture Only**
- **Current**: Linear stack of layers
- **Limitation**: Can't build ResNets, branching architectures, or skip connections
- **Trade-off**: Simplicity vs flexibility
- **Why**: Demonstrates fundamentals; graph-based architectures add significant complexity

#### 5. **Manual Architecture Definition**
- **Current**: List-based layer specification
- **Alternative**: Could implement Keras-like sequential API
- **Trade-off**: Explicit control vs user-friendliness
- **Design Choice**: Transparency for educational value

#### 6. **Limited Activation Function Customization**
- **Current**: Fixed set of activations (ReLU, sigmoid, etc.)
- **Limitation**: Can't add custom activations without modifying source
- **Future**: Add plugin system for custom functions

### Performance vs Accuracy Trade-off

The benchmark results reveal an interesting trade-off:

```
Custom MLP: 4.47x faster training | 97.0% test accuracy
PyTorch:    Baseline speed       | 98.5% test accuracy
```

**Analysis:**
- 1.5% accuracy difference is acceptable for many real-world applications
- 4x speed improvement enables faster experimentation cycles
- For production ML systems with strict latency SLAs, custom implementation may be preferable

**When the trade-off is worth it:**
- Low-latency serving requirements (< 10ms inference)
- Edge deployment with limited compute
- Rapid prototyping phase (faster iterations)

**When it's not:**
- Healthcare/finance where accuracy is critical
- Large-scale training (> 100K samples)
- State-of-the-art model performance needed

---

## Contributing

Contributions are welcome! Areas of interest:
- Regularization techniques
- Additional optimizers (AdaGrad, Nadam)
- Convolutional layers
- Recurrent layers
- Performance optimizations

Please open an issue first to discuss proposed changes.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with the goal of mastering ML fundamentals and showcasing engineering discipline. Inspired by Andrew Ng's Deep Learning Specialization and the need to understand what happens beneath high-level APIs.

---

## Contact

For questions or collaboration opportunities, reach out via GitHub issues or [adhishafwan@gmail.com](adhishafwan@gmail.com).

**⭐ If this project helped you learn, please consider starring the repository!**
