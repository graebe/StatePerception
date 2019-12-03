# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:06:00 2019

@author: P355139
"""

# Imports
from chua_circuit import chua_circuit
import numpy as np
from keras.utils import to_categorical

# State Perception Imports
from StatePerception import SPDataset, SPDatasetPlotter
from StatePerception import SP_WindowedFeedForward_Classifier
from StatePerception import SPModelTrainer

# Data Generation
cc = chua_circuit()
data = cc.simulate_example(steps=10000)
X = data[:, [0, 1, 2]]                      # System Inputs
Y = to_categorical(data[:, [3]]>0)          # System Outputs - Version 1

# Training, Validation and Test Split
X_train = X[:8000, :]
Y_train = Y[:8000, :]
I_train = np.ones(shape=(Y_train.shape[0],1))
I_train[100:250] = 0
data_train = [[[X_train], [Y_train], [I_train]],
              [[X_train], [Y_train], [I_train]]]
X_val = X[8000:9000, :]
Y_val = Y[8000:9000, :]
I_val = np.ones(shape=(Y_val.shape[0],1))
data_val = [[[X_val], [Y_val], [I_val]]]
X_test = X[9000:, :]
Y_test = Y[9000:, :]
I_test = np.ones(shape=(Y_test.shape[0],1))
data_test = [[[X_test], [Y_test], [I_test]]]

# SPDataset Setup
spdata = SPDataset(name='Chua Circuit Classification',
                   data_train=data_train,
                   data_val=data_val,
                   data_test=data_test,
                   dt=0.01,
                   X_signal_names=[['Input 1', 'Input 2', 'Input 3']],
                   Y_signal_names=[['Class 1', 'Class 2']])

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
spmodel = SP_WindowedFeedForward_Classifier(n_features=spdata.n_features,
                                            n_outputs=spdata.n_outputs,
                                            scales_X=spdata.X_scale,
                                            means_X=spdata.X_mean,
                                            scales_Y=None,
                                            means_Y=None
                                            )

# Trainer for Neual Network
sptrainer = SPModelTrainer(spmodel, spdata)

# Fit Neural Network
sptrainer.fit_model(timesteps=100,
                    batch_size=128,
                    epochs=1000,
                    discard_invalid='last timestep')

# Show Training Curves
sptrainer.show_training_curves('loss')

# Show Fit
sptrainer.show_fit(dataset='train', timesteps=100)
sptrainer.show_fit(dataset='val', timesteps=100)

# Show Error Plots
sptrainer.show_error_plot(timesteps=100)

# Freeze Model for Later Deployment
spmodel.freeze_model(savename='chua_circuit_class_freeze',savepath='frozen_model/')
