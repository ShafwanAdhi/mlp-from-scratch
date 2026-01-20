import numpy as np
class mlp:
  def __init__(self):
    self.learning_rate = 0.01
    self.activation_list = {
        'relu':self._relu,
        'leaky_relu':self._leaky_relu,
        'tanh':self._tanh,
        'sigmoid':self._sigmoid,
        'softmax':self._softmax,
        'linear':lambda x:x
    }
    self.der_activation_list = {
        'relu':self._der_relu,
        'leaky_relu':self._der_leaky_relu,
        'tanh':self._der_tanh,
        'sigmoid':self._der_sigmoid,
        'softmax':self._der_softmax,
        'linear':self._der_linear
    }
    self.loss_list = {
        'mse':self._mse,
        'mae':self._mae,
        'bce':self._bce,
        'cce':self._cce
    }
    self.der_loss_list = {
        'mse':self._der_mse,
        'mae':self._der_mae,
        'bce':self._der_sigmoid_bce,
        'cce':self._der_softmax_cce
    }
    self.initialization_activation = {
        'relu': self._he_initialization,
        'leaky_relu': self._he_initialization,
        'tanh': self._xavier_initialization,
        'sigmoid': self._xavier_initialization,
        'softmax': self._xavier_initialization,
        'linear': self._xavier_initialization,
    }
    self.optimizer_list = {
        'sgd':self._sgd,
        'sgd_momentum':self._sgd_momentum,
        'RMSprop':self._RMSprop,
        'adam':self._adam
    }
    return

  #--------ACTIVATION LIST---------
  def _relu(self, X: np.ndarray):
        return np.maximum(0, X)

  def _leaky_relu(self, X: np.ndarray, alpha: float = 0.01):
      return np.where(X > 0, X, alpha * X)

  def _tanh(self, X: np.ndarray):
      return np.tanh(X)

  def _sigmoid(self, X: np.ndarray):
      return 1.0 / (1.0 + np.exp(-X))

  def _softmax(self, X: np.ndarray):
      # numerical stability
      X_shifted = X - np.max(X)
      exp_X = np.exp(X_shifted)
      return exp_X / np.sum(exp_X)

  def _der_linear(self, X):
    return 1

  #--------DERIVATIVE ACTIVATION LIST---------
  def _der_relu(self, Z: np.ndarray):
      return (Z > 0).astype(float)

  def _der_leaky_relu(self, Z: np.ndarray, alpha: float = 0.01):
      dZ = np.ones_like(Z)
      dZ[Z < 0] = alpha
      return dZ

  def _der_tanh(self, A: np.ndarray):
      return 1.0 - A ** 2

  def _der_sigmoid(self, A: np.ndarray):
      return A * (1.0 - A)

  def _der_softmax(self, A: np.ndarray):
      return A * (1.0 - A)

  #--------LOSS FUNCTION LIST---------
  def _mse(self, y_pred: np.ndarray, y_true: np.ndarray):
      return np.mean((y_pred - y_true) ** 2)

  def _mae(self, y_pred: np.ndarray, y_true: np.ndarray):
      return np.mean(np.abs(y_pred - y_true))

  def _bce(self, y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-7):
      y_pred = np.clip(y_pred, eps, 1 - eps)
      bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
      return np.mean(bce)

  def _cce(self, y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-9):
      y_pred = np.clip(y_pred, eps, 1 - eps)
      return -np.sum(y_true * np.log(y_pred))

  #--------DERIVATIVE LOSS FUNCTION LIST---------
  def _der_mse(self, y_pred: np.ndarray, y_true: np.ndarray):
      return 2.0 * (y_pred - y_true) / y_true.size

  def _der_mae(self, y_pred: np.ndarray, y_true: np.ndarray):
      grad = np.zeros_like(y_pred)
      grad[y_pred > y_true] = 1.0
      grad[y_pred < y_true] = -1.0
      return grad / y_true.size

  def _der_sigmoid_bce(self, A: np.ndarray, Y: np.ndarray):
      return A-Y

  def _der_softmax_cce(self, A: np.ndarray, Y: np.ndarray):
      return A-Y

  #--------WEIGHTS INITIALIZATION---------

  def _xavier_initialization(self, fan_in:int, fan_out:int):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

  def _he_initialization(self, fan_in:int, fan_out:int):
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

  #--------INITIALIZATION---------
  def _normalize_architecture(self, architecture:list):
    self.hidden_layers = []
    self.activation = {}
    for i, var in enumerate(architecture):
      if isinstance(var, str):
        self.activation[len(self.hidden_layers)-1] = var
      else:
        self.hidden_layers.append(var)
        self.activation[len(self.hidden_layers)-1] = "linear"

  def _input_validation(self):
    for act in self.activation.values():
      if act not in self.activation_list:
        raise ValueError(f"there is no activation for {act}")
    return

  def _init_velocity_weight_bias(self):
    self.velocity_weight = [np.zeros_like(w) for w in self.weights]
    self.velocity_bias = [np.zeros_like(b) for b in self.bias]
    return

  def _init_velocity_adam(self):
    self.vm_weight = [np.zeros_like(w) for w in self.weights]
    self.vm_bias = [np.zeros_like(b) for b in self.bias]
    self.vr_weight = [np.zeros_like(w) for w in self.weights]
    self.vr_bias = [np.zeros_like(b) for b in self.bias]
    self.adam_t = 0  #timestep counter
    return

  def _init_bias_weights(self):
    self.weights = []
    self.bias = []
    for i in range(len(self.hidden_layers)-1):
      fan_in = self.hidden_layers[i+1]
      fan_out = self.hidden_layers[i]
      activation = self.activation[i+1] if i+1 in self.activation else 'xavier'
      self.weights.append(self.initialization_activation[activation](fan_in, fan_out))
      self.bias.append(np.zeros(fan_in))
    return

  def sequential(self, architecture:list):
    self._normalize_architecture(architecture)
    self._input_validation()
    self._init_bias_weights()
    return

  def _loss_validation(self, loss):
    #cek apakah loss ada di loss_list
    if loss not in self.loss_list:
      raise ValueError('unidentified loss function')

    #cek apakah pasangan activation func terakhir dan loss cocok
    last_act = self.activation[len(self.hidden_layers)-1]
    if loss == "bce" and last_act != 'sigmoid':
      raise ValueError('bce must be with sigmoid')
    if loss == "cce" and last_act != 'softmax':
      raise ValueError('cce must be with softmax')
    return

  def set_loss(self, loss):
    self._loss_validation(loss)
    self.loss = loss
    return

  def calculate_loss(self, predict, label):
    label = np.array(label)
    self.loss_now = self.loss_list[self.loss](predict, label)
    self.label = np.array(label)
    return self.loss_now

  def set_optimizer(self, optimizer):
    self.optimizer = optimizer
    if optimizer == "sgd_momentum" or optimizer == "RMSprop":
      self._init_velocity_weight_bias()
    if optimizer == "adam":
      self._init_velocity_adam()
    return

  #--------FORWARD PROPAGATION---------
  def forward(self, X:list):
    #validasi
    if len(X) != self.hidden_layers[0]:
      raise ValueError("Missmatch input features")
    #inisialisasi
    self.forward_result = {'pre_act': [],
                           'post_act': []}
    X = np.array(X)
    input = X
    self.input = input
    for i, (weight, bias) in enumerate(zip(self.weights, self.bias)):
      result = weight @ input + bias
      input = result
      self.forward_result['pre_act'].append(input)
      if i+1 in self.activation:
        post_act = self.activation_list[self.activation[i+1]](result)
        result = post_act
        input = result
      self.forward_result['post_act'].append(post_act)
    return result

  #--------BACKPROPAGATION---------
  def backprop_zloss(self):
    #find last layer grad
    last_act = self.activation[len(self.hidden_layers)-1]
    preds = self.forward_result['post_act'][-1]
    labels = self.label
    if (last_act == 'sigmoid' and self.loss == 'bce') or (last_act == 'softmax' and self.loss == 'cce'):
      self.gradient['z'].append(np.array(self._der_sigmoid_bce(preds,labels)))
    else:
      loss_grad = self.der_loss_list[self.loss](preds,labels)
      if last_act == "relu" or last_act == "leaky_relu":
        act_grad = self.der_activation_list[last_act](self.forward_result['pre_act'][-1])
      else:
        act_grad = self.der_activation_list[last_act](preds)
      self.gradient['z'].append(np.array(loss_grad * act_grad))

    for i in range(len(self.hidden_layers)-2,0,-1):
      grad_a = (self.weights[i].T @ self.gradient['z'][-1])
      if self.activation[i] == 'relu' or self.activation[i] == 'leaky_relu':
        grad_z = self.der_activation_list[self.activation[i]](self.forward_result['pre_act'][i-1])
      else:
        grad_z = self.der_activation_list[self.activation[i]](self.forward_result['post_act'][i-1])
      result = grad_a * grad_z
      self.gradient['z'].append(result)
    self.gradient['z'] = self.gradient['z'][::-1]
    return

  def backprop_bias(self):
    self.gradient['b'] = self.gradient['z']

  def backprop_weight(self):
    for i in range(len(self.hidden_layers)-2,-1,-1):
      if i-1 != -1:
        result = np.outer(self.gradient['z'][i], self.forward_result['post_act'][i-1])
      else:
        result = np.outer(self.gradient['z'][i], self.input)
      self.gradient['w'].insert(0,result)
    return

  def backprop(self):
    self.gradient = {
        'z': [],
        'b': [],
        'w': []
    }
    self.backprop_zloss()
    self.backprop_bias()
    self.backprop_weight()
    return

  #--------OPTIMIZE---------
  def optimize(self):
    self.optimizer_list[self.optimizer]()

  def _sgd(self):
    #bias
    for i, (grad, bias) in enumerate(zip(self.gradient['b'], self.bias)):
      result = bias - grad * self.learning_rate
      self.bias[i] = result
    #weight
    for i, (grad, weight) in enumerate(zip(self.gradient['w'], self.weights)):
      result = weight - grad * self.learning_rate
      self.weights[i] = result

  def _sgd_momentum(self, beta = 0.9):
    #bias
    for i, (grad, bias) in enumerate(zip(self.gradient['b'], self.bias)):
      self.velocity_bias[i] = beta * self.velocity_bias[i] - self.learning_rate * self.gradient['b'][i]
      self.bias[i] += self.velocity_bias[i]
    #weight
    for i, (grad, weight) in enumerate(zip(self.gradient['w'], self.weights)):
      self.velocity_weight[i] = beta * self.velocity_weight[i] - self.learning_rate * self.gradient['w'][i]
      self.weights[i] += self.velocity_weight[i]
    return

  def _RMSprop(self, beta = 0.9):
    eps = np.finfo(np.float32).eps
    #bias
    for i, (grad, bias) in enumerate(zip(self.gradient['b'], self.bias)):
        self.velocity_bias[i] = beta * self.velocity_bias[i] + ((1 - beta) * (self.gradient['b'][i] ** 2))
        self.bias[i] = self.bias[i] - (self.learning_rate * (self.gradient['b'][i] / np.sqrt(self.velocity_bias[i] + eps)))
    #weight
    for i, (grad, weight) in enumerate(zip(self.gradient['w'], self.weights)):
        self.velocity_weight[i] = beta * self.velocity_weight[i] + ((1 - beta) * (self.gradient['w'][i] ** 2))  # UBAH DARI - MENJADI +
        self.weights[i] = self.weights[i] - (self.learning_rate * (self.gradient['w'][i] / np.sqrt(self.velocity_weight[i] + eps)))
    return

  def _adam(self, beta1=0.9, beta2=0.999):
      eps = 1e-8
      self.adam_t += 1  # increment timestep
      
      # bias
      for i in range(len(self.bias)):
          self.vm_bias[i] = beta1 * self.vm_bias[i] + (1 - beta1) * self.gradient['b'][i]
          self.vr_bias[i] = beta2 * self.vr_bias[i] + (1 - beta2) * (self.gradient['b'][i] ** 2)

          m_hat = self.vm_bias[i] / (1 - beta1 ** self.adam_t)
          v_hat = self.vr_bias[i] / (1 - beta2 ** self.adam_t)

          self.bias[i] = self.bias[i] - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
      
      # weight
      for i in range(len(self.weights)):
          self.vm_weight[i] = beta1 * self.vm_weight[i] + (1 - beta1) * self.gradient['w'][i]
          self.vr_weight[i] = beta2 * self.vr_weight[i] + (1 - beta2) * (self.gradient['w'][i] ** 2)

          m_hat = self.vm_weight[i] / (1 - beta1 ** self.adam_t)
          v_hat = self.vr_weight[i] / (1 - beta2 ** self.adam_t)

          self.weights[i] = self.weights[i] - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
      return

  #--------TRAINING LOOP---------
  def fit(self, X:list, Y:list, epoch, learning_rate=0.005, lr_decay = 1):
    self.learning_rate = learning_rate
    for i in range(epoch):
      for x,y in zip(X,Y):
        result = self.forward(x)
        loss = self.calculate_loss(result, y)
        self.backprop()
        self.optimize()
      self.learning_rate *= lr_decay
      if i % 100 == 0:
        print(f'epoch {i} - loss: {loss}')

    print(f'final loss: {loss}')
    return
