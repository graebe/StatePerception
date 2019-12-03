# Imports
from chua_circuit import chua_circuit
import numpy as np

# SP Imports
from StatePerception import SPDataset
from StatePerception import SPFrozenModel

# Data Generation
cc = chua_circuit()
data = cc.simulate_example(steps=10000)
X = data[:, [0, 1, 2]]                        # System Inputs
Y = data[:, [3]]                              # System Outputs - Version 1

# Training, Validation and Test Split
X_train = X[:8000, :]
Y_train = Y[:8000, :]
I_train = np.ones_like(Y_train)
I_train[100:250] = 0
data_train = [[[X_train], [Y_train], [I_train]],
              [[X_train], [Y_train], [I_train]]]
X_val = X[8000:9000, :]
Y_val = Y[8000:9000, :]
I_val = np.ones_like(Y_val)
data_val = [[[X_val], [Y_val], [I_val]]]
X_test = X[9000:, :]
Y_test = Y[9000:, :]
data_test = [[[X_test], [Y_test], [np.ones_like(Y_test)]]]

# SPDataset Setup
spdata = SPDataset(name='Chua Circuit',
                   data_train=data_train,
                   data_val=data_val,
                   data_test=data_test,
                   dt=0.01,
                   X_signal_names=[['Input 1', 'Input 2', 'Input 3']],
                   Y_signal_names=[['Output 1']])

# Initialize
spfreeze = SPFrozenModel()

# Load Frozen Model
spfreeze.load_frozen_model('chua_circuit_freeze')

# Predict
y_pred = spfreeze.predict([spdata.X_scaled[0][300:400]])

# Check Results
print('Reference: {0:}'.format(spdata.Y_scaled[0][399]))
print('Result of prediction: {0:}'.format(y_pred[0]))
