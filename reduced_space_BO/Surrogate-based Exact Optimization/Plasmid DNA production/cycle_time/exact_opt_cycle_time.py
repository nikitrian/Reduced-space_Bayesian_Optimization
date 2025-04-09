import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import pickle 
import tempfile
from omlt.io import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.neuralnet import ReluBigMFormulation
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'

metric = 'cycle time'


class ReLUNet(torch.nn.Module):
        """
        ReLU neural network structure.
        1 hidden layers, 1 output layer.
        size of input, output, and hidden layers are specified
        """
        def __init__(self, n_input, n_hidden, n_output, num_layers):
            super().__init__()

            self.hidden_layers = torch.nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    self.hidden_layers.append(torch.nn.Linear(n_input, n_hidden))
                else:
                    self.hidden_layers.append(torch.nn.Linear(n_hidden, n_hidden))


            self.output = nn.Linear(n_hidden, n_output)
            self.relu = nn.ReLU()
    
        def forward(self, x):
            for layer in self.hidden_layers:
                x = self.relu(layer(x))
            x = self.output(x)
            return x
        


def load_model(fn, train_set):
    """
    Load ANN model for inference
    """

    P = load_pkl(fn)
    SD = P['state_dict']
    
    structure = P['structure']
    
    
    inputs = train_set[0].shape[1]
    outputs = train_set[1].shape[1]
    m = ReLUNet(inputs, structure[2], outputs, structure[3]).to(torch.float64)
    m.load_state_dict(SD)

    return m


def mm_norm(arr, normP):
    """
    Min max normalisation
    """
    arrmax = normP[0] 
    arrmin = normP[1]
    return (arr - arrmin)/(arrmax - arrmin)

def mm_rev(norm, normP):
    """
    Reverse min max normalisation
    """
    arrmax = normP[0] 
    arrmin = normP[1]
    return norm*(arrmax - arrmin) + arrmin

with open('device.txt', 'w') as f:
    f.write(device)
    f.write(f'\n{torch.cuda.is_available()}')


def load_pkl(file_name):
    """
    Loads .pkl file from path: file_name
    """
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    print(f'Data loaded: {file_name}')
    return data


# Reduced-space decision variables with their lower and upper bounds 

# Original upper and lower bounds used for optimization
space_og = [
    #{'name': 'flask_time', 'type': 'continuous', 'domain': (21600, 108000)},
    {'name': 'seed_time', 'type': 'continuous', 'domain': (43200, 108000)},
    #{'name': 'seed_fed_batch', 'type': 'continuous', 'domain': (0.001636, 0.004908)},
    #{'name': 'seed_batch_med', 'type': 'continuous', 'domain': (0.010225325, 0.030675975)},
    #{'name': 'seed_conversion', 'type': 'continuous', 'domain': (0.9, 0.98)},
    {'name': 'main_time', 'type': 'continuous', 'domain': (64800, 144000)},
    {'name': 'fed_batch', 'type': 'continuous', 'domain': (0.0190875, 0.0572625)},
    {'name': 'batch_med', 'type': 'continuous', 'domain': (0.102253255, 0.306759765)},
    #{'name': 'conversion', 'type': 'continuous', 'domain': (0.9, 0.98)},
    #{'name': 'solid_conc', 'type': 'continuous', 'domain': (50, 300)},
    {'name': 'res_vol', 'type': 'continuous', 'domain': (0.28903585,  0.86710755)},
    #{'name': 'equi1', 'type': 'continuous', 'domain': (0.01, 0.05)},
    {'name': 'dia1', 'type': 'discrete', 'domain': (10, 50)},
    {'name': 'flush1', 'type': 'continuous', 'domain': (0.0015,  0.0045)},
    #{'name': 'equi2', 'type': 'continuous', 'domain': (0.01, 0.05)},
    #{'name': 'dia2', 'type': 'discrete', 'domain': (10, 50)},
    #{'name': 'flush2', 'type': 'continuous', 'domain': (0.00067,  0.002)},
    #{'name': 'failure', 'type': 'continuous', 'domain': (0.0, 0.1)},
]

# Upper and lower bounds used for sampling  up to +- 10% of the original
space = [
    #{'name': 'flask_time', 'type': 'continuous', 'domain': (21600*0.9, 108000*1.1)},
    {'name': 'seed_time', 'type': 'continuous', 'domain': (43200*0.9, 108000*1.1)},
    #{'name': 'seed_fed_batch', 'type': 'continuous', 'domain': (0.001636*0.9, 0.004908*1.1)},
    #{'name': 'seed_batch_med', 'type': 'continuous', 'domain': (0.010225325*0.9, 0.030675975*1.1)},
    #{'name': 'seed_conversion', 'type': 'continuous', 'domain': (0.9*0.9, 0.98*1.1)},
    {'name': 'main_time', 'type': 'continuous', 'domain': (64800*0.9, 144000*1.1)},
    {'name': 'fed_batch', 'type': 'continuous', 'domain': (0.0190875*0.9, 0.0572625*1.1)},
    {'name': 'batch_med', 'type': 'continuous', 'domain': (0.102253255*0.9, 0.306759765*1.1)},
    #{'name': 'conversion', 'type': 'continuous', 'domain': (0.9*0.9, 0.98*1.1)},
    #{'name': 'solid_conc', 'type': 'continuous', 'domain': (50*0.9, 300*1.1)},
    {'name': 'res_vol', 'type': 'continuous', 'domain': (0.28903585*0.9,  0.86710755*1.1)},
    #{'name': 'equi1', 'type': 'continuous', 'domain': (0.01*0.9, 0.05*1.1)},
    {'name': 'dia1', 'type': 'discrete', 'domain': (10*0.9, 50*1.1)},
    {'name': 'flush1', 'type': 'continuous', 'domain': (0.0015*0.9,  0.0045*1.1)},
    #{'name': 'equi2', 'type': 'continuous', 'domain': (0.01*0.9, 0.05*1.1)},
    #{'name': 'dia2', 'type': 'discrete', 'domain': (10*0.9, 50*1.1)},
    #{'name': 'flush2', 'type': 'continuous', 'domain': (0.00067*0.9,  0.002*1.1)},
    #{'name': 'failure', 'type': 'continuous', 'domain': (0.0*0.9, 0.1*1.1)},
]

inputlength = len(space)

# Load inputs and outputs from CSV files
input_data = pd.read_csv('inputs.csv')
output_data = pd.read_csv('outputs.csv')

# Keep only the productivity column
output_data = output_data[[metric]]


X_train, X_test, y_train, y_test = train_test_split(input_data, 
                                                    output_data, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible


# Split data into train and test sets
Xtrain = X_train.to_numpy(dtype = 'float64')
Ytrain = y_train.to_numpy(dtype = 'float64')
Xtest = X_test.to_numpy(dtype = 'float64')
Ytest = y_test.to_numpy(dtype = 'float64')

# Min max normalisation
Xmax = np.array([var['domain'][1] for var in space])
Xmin = np.array([var['domain'][0] for var in space])

XnormP = (Xmax, Xmin)

normP = XnormP

train_set = (mm_norm(Xtrain, normP), Ytrain)
test_set = (mm_norm(Xtest, normP), Ytest)



# Load the trained ANN model
ANN_path = 'ann_ACC_0.35_5884_0.0020_512_1.pkl'
ann_loaded = load_model(ANN_path, train_set)

input_bounds_og = torch.tensor([var['domain'] for var in space_og], dtype=torch.float64)

# Normalize input bounds using normalize_x function
normalized_bounds_low = mm_norm(input_bounds_og[:,0], normP)
normalized_bounds_up = mm_norm(input_bounds_og[:,1], normP)

# Convert input_bounds_og to the format of input_bounds
input_bounds = [(normalized_bounds_low[i].item(), normalized_bounds_up[i].item()) for i in range(len(normalized_bounds_low))]

# model input used for exporting
x = torch.randn(1, inputlength,  requires_grad=True, dtype=torch.float64)
pytorch_model = None
with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
    torch.onnx.export(
        ann_loaded,
        x,
        f,
        input_names= ['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    write_onnx_model_with_bounds(f.name, None, input_bounds)
    print(f"Wrote PyTorch model to {f.name}")
    pytorch_model = f.name


network_definition = load_onnx_neural_network_with_bounds(pytorch_model)

#create a Pyomo model with an OMLT block
model = pyo.ConcreteModel()
model.nn = OmltBlock()

#multiple formulations of a neural network are possible
#this uses the default NeuralNetworkFormulation object
formulation = ReluBigMFormulation(network_definition)

#build the formulation on the OMLT block
model.nn.build_formulation(formulation)

# Objective function
model.obj = pyo.Objective(expr=(model.nn.outputs[0]), sense=pyo.minimize)

# Constraints
model.con = pyo.Constraint(expr=model.nn.inputs[0] <= model.nn.inputs[1])

start_time = time.time()

pyo.SolverFactory('gurobi').solve(model, tee=True, options={'TimeLimit': 57600})

time.sleep(2)  
end_time = time.time()
execution_time= end_time - start_time

print(f"Execution time: {execution_time}")

# Get the opimal values for the decision variables, then plug them in the simulation model to get the final value of the objective function
def get_optimized_input(model, normP):
    x_opt = torch.tensor(list(model.nn.inputs.extract_values().values()), dtype=torch.float64).unsqueeze(0)
    return mm_rev(x_opt, normP)


print(get_optimized_input(model, normP))


