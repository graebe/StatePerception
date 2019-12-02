# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:50:11 2019

Testing SPDataset

@author: Torben Gräber
"""

###############################################################################
# Imports
import numpy as np
import sys
sys.path.insert(0,'../')
from StatePerception import SPDataset, SPDatasetPlotter

def merge_dicts(dicts): 
    if dicts is not(None) and len(dicts)>0:
        if len(dicts)==1:
            dict_joined = dicts[0].copy()
            return dict_joined
        elif len(dicts)==2:
            dict1 = dicts[0].copy()
            dict2 = dicts[1].copy()
            dict1.update(dict2)
            return dict1
        else:
            dicts = [d.copy() for d in dicts]
            return merge_dicts([merge_dicts([dicts[0],dicts[1]]) , *dicts[2:] ])
    else:
        return None

###############################################################################
# Generate Sample Data for Testing
name = 'dummy name'
dt = 0.01
X_signal_names = ['Xsig1','Xsig2','Xsig3']
Y_signal_names = ['Ysig1','Ysig2']
X_signal_units = ['Xunit1','Xunit2','Xunit3']
Y_signal_units = ['Yunit1','Yunit2']

# Artificial Data Set 1 - One Input Tensor Dim 1, One Output Tensor Dim 1, Two Meas, All Valid
D1_name = 'D1'
D1_X11 = [np.expand_dims(np.arange(0,100),axis=1)]
D1_Y11 = [2*np.expand_dims(np.arange(0,100),axis=1)]
D1_I11 = [np.ones_like(D1_X11[0])]
D1_X12 = [np.expand_dims(np.arange(0,100),axis=1)]
D1_Y12 = [2*np.expand_dims(np.arange(0,100),axis=1)]
D1_I12 = [np.ones_like(D1_X11[0])]
D1_X_train = [np.concatenate([D1_X11,D1_X12],axis=1)]
D1_Y_train = [np.concatenate([D1_Y11,D1_Y12],axis=1)]
D1_I_train = [np.concatenate([D1_I11,D1_I12],axis=1)]
D1_X_val = D1_X_train
D1_Y_val = D1_Y_train
D1_I_val = D1_I_train
D1_X_test = D1_X_train
D1_Y_test = D1_Y_train
D1_I_test = D1_I_train
D1_X = [np.concatenate([D1_X_train[0],D1_X_val[0],D1_X_test[0]],axis=1)]
D1_Y = [np.concatenate([D1_Y_train[0],D1_Y_val[0],D1_Y_test[0]],axis=1)]
D1_I = [np.concatenate([D1_I_train[0],D1_I_val[0],D1_I_test[0]],axis=1)]
D1_train = [[D1_X11,D1_Y11,D1_I11],[D1_X12,D1_Y12,D1_I12]]
D1_val = D1_train
D1_test = D1_train
D1_X_signal_names = [['Xsig1']]
D1_Y_signal_names = [['Ysig1']]
D1_X_signal_units = [['Xunit1']]
D1_Y_signal_units = [['Yunit1']]
D1_dt = 0.01

# Test Meta Data D1
Test_Metadata_D1 = {}
Test_Metadata_D1['name'] = {'target':D1_name}
Test_Metadata_D1['X_signal_names'] = {'target':D1_X_signal_names}
Test_Metadata_D1['Y_signal_names'] = {'target':D1_Y_signal_names}
Test_Metadata_D1['X_signal_units'] = {'target':D1_X_signal_units}
Test_Metadata_D1['Y_signal_units'] = {'target':D1_Y_signal_units}
Test_Metadata_D1['dt'] = {'target':D1_dt}

# Test Data D1
Test_Data_D1 = {}
Test_Data_D1['data_train'] = {'target':D1_train}
Test_Data_D1['data_val'] = {'target':D1_val}
Test_Data_D1['data_test'] = {'target':D1_test}
Test_Data_D1['X'] = {'target':D1_X}
Test_Data_D1['Y'] = {'target':D1_Y}
Test_Data_D1['I'] = {'target':D1_I}
Test_Data_D1['X_train'] = {'target':D1_X_train}
Test_Data_D1['Y_train'] = {'target':D1_Y_train}
Test_Data_D1['I_train'] = {'target':D1_I_train}
Test_Data_D1['X_val'] = {'target':D1_X_val}
Test_Data_D1['Y_val'] = {'target':D1_Y_val}
Test_Data_D1['I_val'] = {'target':D1_I_val}
Test_Data_D1['X_test'] = {'target':D1_X_test}
Test_Data_D1['Y_test'] = {'target':D1_Y_test}
Test_Data_D1['I_test'] = {'target':D1_I_test}

# Artificial Data Set 2 - One Input Tensor Dim 2, One Output Tensor Dim 2, Two Meas, All Valid
X21 = [np.multiply(np.array([[1],[2]]),np.tile(np.arange(0,100),reps=[2,1])).transpose(),2*np.multiply(np.array([[1],[2]]),np.tile(np.arange(0,100),reps=[2,1])).transpose()]
Y21 = [3*np.multiply(np.array([[1],[2]]),np.tile(np.arange(0,100),reps=[2,1])).transpose(),4*np.multiply(np.array([[1],[2]]),np.tile(np.arange(0,100),reps=[2,1])).transpose()]
I21 = [np.ones_like(X21[0]),np.ones_like(X21[1])]
X22 = [5*np.multiply(np.array([[1],[2]]),np.tile(np.arange(0,100),reps=[2,1])).transpose(),6*np.multiply(np.array([[1],[2]]),np.tile(np.arange(0,100),reps=[2,1])).transpose()]
Y22 = [7*np.multiply(np.array([[1],[2]]),np.tile(np.arange(0,100),reps=[2,1])).transpose(),8*np.multiply(np.array([[1],[2]]),np.tile(np.arange(0,100),reps=[2,1])).transpose()]
I22 = [np.ones_like(X21[0]),np.ones_like(X21[1])]
D2_train = [[X21,Y21,I21],[X22,Y22,I22]]
D2_val = D2_train
D2_test = D2_train
D2_X_signal_names = [['X1sig1','X1sig2'],['X2sig1','X2sig2']]
D2_Y_signal_names = [['Y1sig1','Y1sig2'],['Y2sig1','Y2sig2']]
D2_X_signal_units = [['X1unit1','X1unit2'],['X2unit1','X2unit2']]
D2_Y_signal_units = [['Y1unit1','Y1unit2'],['Y2unit1','Y2unit2']]
D2_dt = 0.01

###############################################################################
# Testing Functionalities

def get_test(target):
    if target is None:
        test = 'check_none'
    elif type(target) is bool:
        test = 'direct_comparison'
    elif type(target) is str:
        test = 'direct_comparison'
    elif type(target) is float:
        test = 'direct_comparison'
    elif type(target) is np.ndarray:
        test = 'direct_comparison'
    elif type(target) is list:
        if len(target)==0:
            test = 'direct_comparison'
        else:
            test = 'check_list'
    else:
        test = None
    return test

def check_none(none1,none2):
    return none1 is None and none2 is None

def direct_comparison(bool1,bool2):
    return bool1 == bool2

def check_list(list1,list2):
    if len(list1) == len(list2):
        test_ok = []
        for el1,el2 in zip(list1,list2):
            test_el = get_test(el1)
            test_ok.append(do_test(test_el,el1,el2))
    else:
        test_ok = False
    return np.all(test_ok)

def do_test(test,target,result):
    test_ok = False
    if test == 'check_none':
        test_ok = check_none(target,result)
    elif test == 'direct_comparison':
        test_ok = direct_comparison(target,result)
    elif test == 'check_list':
        test_ok = check_list(target,result)
    return test_ok

def perform_property_tests(testname,Test_curr,spdata,verbose=False):
    print(' ')
    print('Performing ' + testname)
    print('---------------------------------------------------------')
    # Test command
    if verbose:
        print(' ')
        print('Test Summary:')
    for testdef in Test_curr:
        # Get Result
        Test_curr[testdef]['result'] = getattr(spdata, testdef)
        # Get Test
        test = get_test(Test_curr[testdef]['target'])
        # Do test
        Test_curr[testdef]['test_ok'] = do_test(test,Test_curr[testdef]['target'],Test_curr[testdef]['result'])
        # Print if Verbose
        if verbose:
            print('   {0:20}{1:30}'.format(testdef,str(Test_curr[testdef])))
    all_tests_ok = all([Test_curr[key]['test_ok'] for key in Test_curr.keys()])
    if verbose:
        print(' ')
    print('Test Results:')
    if all_tests_ok:
        print('   All tests ok.')
    else:
        print('   WARNING: Tests partially failed. Listing failed Tests:')
        for testdef in Test_curr:
            if Test_curr[testdef]['test_ok']==False:
                if verbose:
                    print('      {0:20}{1:30}'.format(testdef,str(Test_curr[testdef])))
                else:
                    print('      {0:20}'.format(testdef))
    print('---------------------------------------------------------')
    print(' ')

###############################################################################
# Define Standard Property Tets

# Test Meta Data Empty
Test_Metadata_Empty = {}
Test_Metadata_Empty['name'] = {'target':None}
Test_Metadata_Empty['X_signal_names'] = {'target':None}
Test_Metadata_Empty['Y_signal_names'] = {'target':None}
Test_Metadata_Empty['X_signal_units'] = {'target':None}
Test_Metadata_Empty['Y_signal_units'] = {'target':None}
Test_Metadata_Empty['dt'] = {'target':None}

# Test Meta Data Filled
Test_Metadata_Filled = {}
Test_Metadata_Filled['name'] = {'target':name}
Test_Metadata_Filled['X_signal_names'] = {'target':X_signal_names}
Test_Metadata_Filled['Y_signal_names'] = {'target':Y_signal_names}
Test_Metadata_Filled['X_signal_units'] = {'target':X_signal_units}
Test_Metadata_Filled['Y_signal_units'] = {'target':Y_signal_units}
Test_Metadata_Filled['dt'] = {'target':dt}

# Test Data Empty
Test_Data_Empty = {}
Test_Data_Empty['data_train'] = {'target':None}
Test_Data_Empty['data_val'] = {'target':None}
Test_Data_Empty['data_test'] = {'target':None}
Test_Data_Empty['X'] = {'target':None}
Test_Data_Empty['Y'] = {'target':None}
Test_Data_Empty['I'] = {'target':None}
Test_Data_Empty['X_train'] = {'target':None}
Test_Data_Empty['Y_train'] = {'target':None}
Test_Data_Empty['I_train'] = {'target':None}
Test_Data_Empty['X_val'] = {'target':None}
Test_Data_Empty['Y_val'] = {'target':None}
Test_Data_Empty['I_val'] = {'target':None}
Test_Data_Empty['X_test'] = {'target':None}
Test_Data_Empty['Y_test'] = {'target':None}
Test_Data_Empty['I_test'] = {'target':None}

# Test Scaling
Test_Scaler_Empty = {}
Test_Scaler_Empty['X_scaler'] = {'target':None}
Test_Scaler_Empty['Y_scaler'] = {'target':None}

# Test Miscellaneous
Test_Misc_Standard = {}
Test_Misc_Standard['verbose'] = {'target':True}

###############################################################################
# Define Standard Function Tets

def perform_function_tests(testname,Test_curr,spdata):
    print(' ')
    print('Performing ' + testname)

    print(' ')

# Test Get Tensor
Test_Get_Tensor = {}
Test_Get_Tensor = {'function':'_get_tensor','arguments':{},'target':0}

###############################################################################
# Test1 - Initializing empty datasets
testname1 = 'Test - Empty Initialization'
# Pick Tests
Test1 = merge_dicts([Test_Metadata_Empty,Test_Data_Empty,Test_Scaler_Empty,Test_Misc_Standard])
# Executing Command
spdata1 = SPDataset()
# Perform Test
perform_property_tests(testname1,Test1,spdata1)

###############################################################################
# Test2 - Initializing empty datasets with Meta Data
testname2 = 'Test - Data Empty - Metadata Filled'
# Pick Tests
Test2 = merge_dicts([Test_Metadata_Filled,Test_Data_Empty,Test_Scaler_Empty,Test_Misc_Standard])
# Executing command
spdata2 = SPDataset(name=name,
                   data_train=None,
                   data_val=None,
                   data_test=None,
                   dt=dt,
                   X_signal_names=X_signal_names,
                   Y_signal_names=Y_signal_names,
                   X_signal_units=X_signal_units,
                   Y_signal_units=Y_signal_units)
# Perform Test
perform_property_tests(testname2,Test2,spdata2)

###############################################################################
# Test3 - Plotter
testname3 = 'Test - Standard Functionality on Dummy Data Set D1'
# Pick Tests
Test3 = merge_dicts([Test_Metadata_D1,Test_Data_D1])
# Executing command
spdata3 = SPDataset(name=D1_name,
                   data_train=D1_train,
                   data_val=D1_train,
                   data_test=D1_train,
                   dt=D1_dt,
                   X_signal_names=D1_X_signal_names,
                   Y_signal_names=D1_Y_signal_names,
                   X_signal_units=D1_X_signal_units,
                   Y_signal_units=D1_Y_signal_units)
spp3 = SPDatasetPlotter(spdata3)
# Property Tests
perform_property_tests(testname3,Test3,spdata3)
# Plotting Tests
if False:
    spdata3.summary()
    spp3.plot_dataset(dataset='train')
    spp3.plot_dataset(dataset='val')
    spp3.plot_dataset(dataset='test')
    spp3.plot_dataset(dataset='train',plot_X=False)
    spp3.plot_dataset(dataset='train',plot_Y=False)
    spp3.plot_dataset(dataset='train',plot_legend=False)
    spp3.plot_dataset(dataset='train',sharex=False)
    spp3.plot_dataset(dataset='train',plot_X=False,sharex=False)
    spp3.plot_dataset(dataset='train',plot_Y=False,sharex=False)
    spp3.plot_dataset(dataset='train',downsample=10)
    spp3.plot_dataset(dataset='train',scaled_X=True,scaled_Y=False)
    spp3.plot_dataset(dataset='train',scaled_X=False,scaled_Y=True)
    spp3.plot_dataset(dataset='train',scaled_X=True,scaled_Y=True)
    spp3.plot_dataset(dataset='train',indices=[50,150])
    spp3.plot_dataset(dataset='train',pick_signals_X=[[0]],pick_signals_Y=[[0]])

###############################################################################
# Test4 - Plotter
testname4 = 'Test - Standard Functionality on Multidimensional Dummy Data'
# Pick Tests
Test4 = merge_dicts([Test_Metadata_Filled,Test_Data_Empty,Test_Scaler_Empty,Test_Misc_Standard])
# Executing command
spdata4 = SPDataset(name=name,
                   data_train=D2_train,
                   data_val=D2_train,
                   data_test=D2_train,
                   dt=D2_dt,
                   X_signal_names=D2_X_signal_names,
                   Y_signal_names=D2_Y_signal_names,
                   X_signal_units=D2_X_signal_units,
                   Y_signal_units=D2_Y_signal_units)
spp4 = SPDatasetPlotter(spdata4)
# Tests
#spdata4.summary()
if False:
    spp4.plot_dataset(dataset='train')
    spp4.plot_dataset(dataset='val')
    spp4.plot_dataset(dataset='test')
    spp4.plot_dataset(dataset='train',plot_X=False)
    spp4.plot_dataset(dataset='train',plot_Y=False)
    spp4.plot_dataset(dataset='train',plot_legend=False)
    spp4.plot_dataset(dataset='train',sharex=False)
    spp4.plot_dataset(dataset='train',plot_X=False,sharex=False)
    spp4.plot_dataset(dataset='train',plot_Y=False,sharex=False)
    spp4.plot_dataset(dataset='train',downsample=10)
    spp4.plot_dataset(dataset='train',scaled_X=True,scaled_Y=False)
    spp4.plot_dataset(dataset='train',scaled_X=False,scaled_Y=True)
    spp4.plot_dataset(dataset='train',scaled_X=True,scaled_Y=True)
    spp4.plot_dataset(dataset='train',indices=[50,150])
    spp4.plot_dataset(dataset='train',pick_signals_X=[[0,1],[0,1]],pick_signals_Y=[[0,1],[0,1]])
    spp4.plot_dataset(dataset='train',pick_signals_X=[[1],[1]],pick_signals_Y=[[1],[1]])
if False:
    spp4.plot_scatter(dataset='all')
    spp4.plot_scatter(dataset='train')
    spp4.plot_scatter(dataset='val')
    spp4.plot_scatter(dataset='test')
    spp4.plot_scatter(dataset='all',tensor_axis_1=['X',1],tensor_axis_2=['X',0])
    spp4.plot_scatter(dataset='all',tensor_axis_1=['X',1],tensor_axis_2=['X',0],scaled_axis_1=True,scaled_axis_2=True)
    spp4.plot_scatter(dataset='all',tensor_axis_1=['X',1],tensor_axis_2=['X',0],indices=[50,150])
    spp4.plot_scatter(dataset='all',tensor_axis_1=['X',1],signal_axis_1=0,signal_axis_2=1,indices=[50,150])

spp4.plot_heatmap(cmap='Reds')
#spp4.plot_heatmap(log=True)
#spp4.plot_heatmap(density=True)
