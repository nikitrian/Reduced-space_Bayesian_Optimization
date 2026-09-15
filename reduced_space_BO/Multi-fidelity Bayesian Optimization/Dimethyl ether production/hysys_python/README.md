# hysys_python
A collection of functions and classes allowing me to perform TEA on hysys simulations.

## hysys_object_persistence
use this to create a json file containing all the relevant information from the hysys simulation

First to generate the json file with the same name as the hysys file. Note that the relevant hysys simulation must be open in Aspen HYSYS.
```python
import hysys_object_persistence as hop

file = 'hysys_file.hsc'
hop.create_json(file)
```

To load the json file back as an object, use the following:
```python
import hysys_object_persistence as hop

file = 'hysys_file.json'
hysys_object = hop.load_json(file)
```

## hysys_tea
do something similar to the following (either works directly on hysys simulation or json file):
```python
import hysys_tea as ht

# Initialise cost curve values and define cepci
cc = ht.cost_curves()
cepci = 576.4

df = ht.unit_pec(hysys_object, cc, cepci)
tot = ht.capex(df, "fs")
acc = ht.acc(tot, 20, 0.1)
```

