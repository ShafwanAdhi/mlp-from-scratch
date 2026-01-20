import pytest
import numpy as np
from mlp_scratch import MLP

class TestMLPBasic:
    def test_initialization(self):
        model = MLP()
        assert model.learning_rate == 0.01
        assert 'relu' in model.activation_list
        assert 'mse' in model.loss_list
    
    def test_sequential_architecture(self):
        model = MLP()
        model.sequential([2, 4, 'relu', 1, 'sigmoid'])
        assert model.hidden_layers == [2, 4, 1]
        assert model.activation[1] == 'relu'
        assert model.activation[2] == 'sigmoid'
    
    def test_forward_pass(self):
        model = MLP()
        model.sequential([2, 3, 'relu', 1, 'sigmoid'])
        model.set_loss('bce')
        output = model.forward([1, 2])
        assert output.shape == (1,)
        assert 0 <= output[0] <= 1  # sigmoid output range
    
    def test_xor_learning(self):
        model = MLP()
        model.sequential([2, 4, 'relu', 1, 'sigmoid'])
        model.set_loss('bce')
        model.set_optimizer('adam')
        
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [[0], [1], [1], [0]]
        
        model.fit(X, y, epoch=1000, learning_rate=0.01)
        
        # Test if model learned XOR
        predictions = [model.forward(x)[0] for x in X]
        assert predictions[0] < 0.5  # 0 XOR 0 = 0
        assert predictions[1] > 0.5  # 0 XOR 1 = 1
        assert predictions[2] > 0.5  # 1 XOR 0 = 1
        assert predictions[3] < 0.5  # 1 XOR 1 = 0

class TestActivations:
    def test_relu(self):
        model = MLP()
        result = model._relu(np.array([-1, 0, 1]))
        np.testing.assert_array_equal(result, np.array([0, 0, 1]))
    
    def test_sigmoid(self):
        model = MLP()
        result = model._sigmoid(np.array([0]))
        assert np.isclose(result[0], 0.5)
    
    def test_softmax(self):
        model = MLP()
        result = model._softmax(np.array([1, 2, 3]))
        assert np.isclose(np.sum(result), 1.0)
        assert np.all(result >= 0) and np.all(result <= 1)

class TestLossFunctions:
    def test_mse(self):
        model = MLP()
        loss = model._mse(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert np.isclose(loss, 0.0)
    
    def test_bce(self):
        model = MLP()
        loss = model._bce(np.array([0.5]), np.array([1]))
        assert loss > 0

if __name__ == "__main__":
    pytest.main([__file__])
