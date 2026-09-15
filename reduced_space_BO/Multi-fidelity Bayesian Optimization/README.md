# Multi-fidelity Bayesian Optimization

This folder contains the multi-fidelity Bayesian optimization scripts. The high-fidelity evaluations call the process simulators, while the low-fidelity evaluations use trained ANN surrogates.

## Structure

- `Plasmid DNA production/`: SuperPro Designer high-fidelity evaluations with ANN low-fidelity models.
- `Dimethyl ether production/`: Aspen HYSYS high-fidelity evaluations with ANN low-fidelity models.
- `Dimethyl ether production/hysys_python/`: shared Aspen/HYSYS helper code used by the DME objectives.
- `requirements.txt`: Python package dependencies for these scripts.

Each objective folder contains:

- `BO_*.py`: multi-fidelity optimization script.
- `inputs.csv` and `outputs.csv`: historical simulator data used by the ANN loader and warm starts.
- `top_inputs.csv` and `top_outputs.csv`: optional high-fidelity warm-start points.
- `ann_*.pkl`: trained ANN surrogates.

The SuperPro folders also include `pDNA.xlsm`, which is used by the Python/Excel COM interface.

## Running

Install the Python dependencies from `requirements.txt` in a Windows Python environment with COM support. The high-fidelity models additionally require local installations and licenses for the relevant simulator:

- SuperPro Designer for the plasmid DNA case.
- Aspen HYSYS for the DME case.

Run an objective from its folder, for example:

 
cd "reduced_space_BO\Multi-fidelity Bayesian Optimization\Plasmid DNA production\productivity"
python BO_productivity.py


The Aspen scripts default to the HYSYS model tracked in `Data generation/Dimethyl ether - Aspen HYSYS/i-dme-complete-gsa-equil.hsc`. Override this with the objective-specific environment variable if needed:

- `CAPEX_HYSYS_FILE`
- `CARBON_HYSYS_FILE`
- `ENERGY_HYSYS_FILE`
- `FLOWRATE_HYSYS_FILE`
- `OPEX_HYSYS_FILE`

Generated optimization outputs are written to `results_*` folders, which are ignored by Git.
