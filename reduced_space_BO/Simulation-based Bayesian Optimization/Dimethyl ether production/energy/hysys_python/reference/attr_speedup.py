# getting shorted attribure list for aspen

import os
import hysys_object_persistence as hop
from pprint import pprint
import time
import jsonpickle

os.system('cls')

# file = 'CH4_20221003.hsc'
file = 'Flash_Pyrolysis.hsc'
# file = 'Bioethanol - 20220803.hsc'


sim = hop.hysys_connection(file)

st = sim.flowsheet.MaterialStreams('14')
attr_list = dir(st)

new_list = []

for i in range(0,len(attr_list)):
    # if attr_list[i] == attr_list[i+1][0:-5]:
    if attr_list[i] + 'Value' in attr_list:
        continue
    elif attr_list[i][0:2] == 'BO':
        continue
    elif attr_list[i][0:3] in ['Set','Get']:
        continue
    elif attr_list[i][0] == '_':
        continue
    else:
        new_list.append(attr_list[i])

print(len(new_list))

material = jsonpickle.encode(new_list)
with open('attr_list_material.json', "w") as f:
    f.write(material)


# -----------------------------------------------
# Repeat for energy Streams
# -----------------------------------------------
en = sim.Flowsheet.EnergyStreams('Q_E-100')
attr_list = dir(en)
new_list = []

for i in range(0,len(attr_list)):
    if attr_list[i] in ['AttachedOpers','PowerValue']:
        new_list.append(attr_list[i])

# print(len((new_list)))

energy = jsonpickle.encode(new_list)
with open('attr_list_energy.json', "w") as f:
    f.write(energy)

