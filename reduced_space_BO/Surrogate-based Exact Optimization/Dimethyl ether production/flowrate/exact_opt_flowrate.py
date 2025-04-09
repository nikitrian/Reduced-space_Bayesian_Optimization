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

metric = 'DME flowrate'

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
elec_min = 24.70 # GBP/MWh
elec_max = 580.55 # GBP/MWh
elec_min *= 0.804354 # USD/MWh
elec_max *= 0.804354 # USD/MWh
elec_min /= 1000 # USD/kWh
elec_max /= 1000 # USD/kWh

elec_nom = 89.2 # GBP/MWh - 02 Jan 2024
elec_nom *= 0.804354 # USD/MWh
elec_nom /= 1000 # USD/kWh

# For the nominal/range of the MeOH reactor size from van Dal 2013
ref_co2_flow = 88e3 # kg/h CO2
co2_flow = 28333.3620500565 # kg/h in simulation
co2_ratio = co2_flow/ref_co2_flow
ref_cat_mass = 44500 # kg catalyst
cat_mass = ref_cat_mass*co2_ratio # kg catalyst in simulation
void = 0.5
density = 1775
meoh_nominal_vol = cat_mass * (1/density) * (1/(1-void)) # m3


# Original upper and lower bounds used for optimization
space_og = [
    {'name': 'H2 ratio', 'type': 'continuous', 'domain': (2.4, 3.6)},                                       # 1
    {'name': 'P_meoh', 'type': 'continuous', 'domain': (5000, 10000)},                                     # 2
    {'name': 'T_meoh_feed', 'type': 'continuous', 'domain': (210, 270)},                                    # 3 
    {'name': 'Meoh_is_ad', 'type': 'continuous', 'domain': (0, 1)},                                        # 4
    {'name': 'rec ratio', 'type': 'continuous', 'domain': (0.94, 0.995)},                                   # 5
    {'name': 'V_meoh', 'type': 'continuous', 'domain': (0.8*meoh_nominal_vol, 1.2*meoh_nominal_vol)},      # 6  
    #{'name': 'T_dme_feed', 'type': 'continuous', 'domain': (250, 300)},                                    # 7
    #{'name': 'P_dme', 'type': 'continuous', 'domain': (1000, 2000)},                                       # 8
    #{'name': 'meoh_vap_frac', 'type': 'continuous', 'domain': (0,1)},                                      # 9
    #{'name': 'dme_vap_frac', 'type': 'continuous', 'domain': (0,1)},                                       # 10
    #{'name': 'trays1', 'type': 'continuous', 'domain': (57*0.8, 57*1.2)},                                   # 11
    #{'name': 'trays2', 'type': 'continuous', 'domain': (17*0.8, 17*1.2)},                                  # 12
    #{'name': 'feed_loc1', 'type': 'discrete', 'domain': (44/57*0.8, 44/57*1.2)},                            # 13
    #{'name': 'feed_loc2', 'type': 'continuous', 'domain': (10/17*0.8, 10/17*1.2)},                         # 14
]

# Upper and lower bounds used for sampling  up to +- 10% of the original
space = [
    {'name': 'H2 ratio', 'type': 'continuous', 'domain': (2.4*0.9, 3.6*1.1)},                                   # 1
    {'name': 'P_meoh', 'type': 'continuous', 'domain': (5000*0.9, 10000*1.1)},                                  # 2
    {'name': 'T_meoh_feed', 'type': 'continuous', 'domain': (210*0.9, 270*1.1)},                                # 3
   {'name': 'Meoh_is_ad', 'type': 'continuous', 'domain': (0, 1)},                                             # 4
    {'name': 'rec ratio', 'type': 'continuous', 'domain': (0.94, 0.995)},                                       # 5
    {'name': 'V_meoh', 'type': 'continuous', 'domain': (0.8*meoh_nominal_vol*0.9, 1.2*meoh_nominal_vol*1.1)},  # 6
    #{'name': 'T_dme_feed', 'type': 'continuous', 'domain': (250*0.9, 300*1.1)},                                # 7
    #{'name': 'P_dme', 'type': 'continuous', 'domain': (1000*0.9, 2000*1.1)},                                   # 8             
    #{'name': 'meoh_vap_frac', 'type': 'continuous', 'domain': (0,1)},                                          # 9
    #{'name': 'dme_vap_frac', 'type': 'continuous', 'domain': (0,1)},                                           # 10            
   #{'name': 'trays1', 'type': 'continuous', 'domain': (57*0.8*0.9, 57*1.2*1.1)},                              # 11
    #{'name': 'trays2', 'type': 'continuous', 'domain': (17*0.8*0.9, 17*1.2*1.1)},                              # 12           
    #{'name': 'feed_loc1', 'type': 'discrete', 'domain': (44/57*0.8, 44/57*1.2)},                               # 13
    #{'name': 'feed_loc2', 'type': 'continuous', 'domain': (10/17*0.8, 10/17*1.2)},                             # 14
]


inputlength = len(space)


# Load inputs and outputs from CSV files
input_data = pd.read_csv('inputs.csv')
output_data = pd.read_csv('outputs.csv')

# Keep only the target metric
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
ANN_path = 'ann_ACC_0.13_44720_0.0002_418_1.pkl'
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

model.obj = pyo.Objective(expr=(model.nn.outputs[0]), sense=pyo.maximize)

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


