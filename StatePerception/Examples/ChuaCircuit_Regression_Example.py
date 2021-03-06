# Imports
from chua_circuit import chua_circuit
import numpy as np

# State Perception Imports
from StatePerception import SPDataset, SPDatasetPlotter
from StatePerception import SP_WindowedFeedForward
from StatePerception import SPModelTrainer

# Data Generation
cc = chua_circuit()
data = cc.simulate_example(steps=10000)
X = data[:, [0, 1, 2]]                        # System Inputs
Y = data[:, [3]]                              # System Outputs - Version 1
Z = data[:, [4]]                              # System Outputs - Version 2

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

# Plotting Tool
spp = SPDatasetPlotter(spdata)
spp.plot_dataset(dataset='all', scaled_X=True, mark_invalid=True)
spp.plot_distribution()
spp.plot_scatter(tensor_axis_1=['X', 0],
                 tensor_axis_2=['X', 0],
                 signal_axis_1=0,
                 signal_axis_2=1,
                 alpha=0.1)
spp.plot_heatmap(tensor_axis_1=['X', 0],
                 tensor_axis_2=['X', 0],
                 signal_axis_1=0,
                 signal_axis_2=1,
                 bins=(100, 100))

# Neural Network for Regression
spmodel = SP_WindowedFeedForward(n_features=[3],
                                 n_outputs=[1],
                                 scales_X=spdata.X_scale,
                                 means_X=spdata.X_mean,
                                 scales_Y=spdata.Y_scale,
                                 means_Y=spdata.Y_mean)

# Trainer for Neual Network
sptrainer = SPModelTrainer(spmodel, spdata)

# Fit Neural Network
sptrainer.fit_model(timesteps=100,
                    batch_size=128,
                    epochs=10000,
                    discard_invalid='last timestep')

# Show Training Curves
sptrainer.show_training_curves('loss')

# Show Fit
sptrainer.show_fit(dataset='train', timesteps=100)
sptrainer.show_fit(dataset='val', timesteps=100)

# Show Error Plots
sptrainer.show_error_plot(timesteps=100)

# Freeze Model for Later Deployment
spmodel.freeze_model(savename='chua_circuit_freeze',savepath='frozen_model/')
