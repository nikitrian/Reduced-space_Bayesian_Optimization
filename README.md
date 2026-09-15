# Reduced-space Bayesian Optimization for Process Flowsheet Optimization

Niki Triantafyllou, Ben Lyons, Andrea Bernardi, Benoit Chachuat, Cleo Kontoravdi, and Maria M. Papathanasiou

This repository contains research code for reduced-space Bayesian optimization of simulator-based process flowsheets. It includes the official implementation for our 2024 *Computers & Chemical Engineering* paper, together with an extension for multi-fidelity Bayesian optimization.

The methods combine process simulators, global sensitivity analysis (GSA), dimensionality reduction, Bayesian optimization, and artificial neural network (ANN) surrogate models.

## Case Studies

- Plasmid DNA production using SuperPro Designer.
- Dimethyl ether (DME) production using Aspen HYSYS.

## Repository Contents

- `Data generation/`: simulator interface files and scripts used to generate data from SuperPro Designer and Aspen HYSYS.
- `GSA/`: global sensitivity analysis inputs, outputs, and metamodel files.
- `reduced_space_BO/Simulation-based Bayesian Optimization/`: reduced-space Bayesian optimization using direct high-fidelity simulator calls.
- `reduced_space_BO/Surrogate-based Bayesian Optimization/`: Bayesian optimization using trained surrogate models.
- `reduced_space_BO/Surrogate-based Exact Optimization/`: exact optimization of ANN surrogates using OMLT.
- `reduced_space_BO/Multi-fidelity Bayesian Optimization/`: multi-fidelity Bayesian optimization using simulator evaluations as high fidelity and ANN surrogates as low fidelity.

## Requirements

Most Python-only analysis files can be inspected without commercial software. Full reproduction of the simulator-based workflows requires a Windows machine with the relevant licensed simulators installed:

- Python 3.10 or 3.11.
- Microsoft Excel with macro support.
- SuperPro Designer for the plasmid DNA case study.
- Aspen HYSYS for the DME case study.
- The Python packages listed in the relevant environment or requirements files.

For the DME/Aspen workflows, an example conda environment is provided in:

```powershell
Data generation\Dimethyl ether - Aspen HYSYS\environment.yml
```

For the multi-fidelity scripts, install the dependencies from:

```powershell
reduced_space_BO\Multi-fidelity Bayesian Optimization\requirements.txt
```

For example:

```powershell
python -m pip install -r "reduced_space_BO\Multi-fidelity Bayesian Optimization\requirements.txt"
```

PyTorch installation can depend on the local CPU/GPU setup, so users may need to follow the official PyTorch installation command for their machine before installing the remaining packages.

## Running the Code

The simulator-based notebooks and scripts should be run from the objective folder that contains the relevant simulator files. For example, the plasmid DNA folders contain both:

- `pDNA.xlsm`: Excel/VBA interface workbook.
- `pDNA.spf`: SuperPro Designer simulation file.

Example simulation-based plasmid DNA workflow:

```powershell
cd "reduced_space_BO\Simulation-based Bayesian Optimization\Plasmid DNA production\productivity"
jupyter notebook plasmid_productivity.ipynb
```

Example multi-fidelity plasmid DNA workflow:

```powershell
cd "reduced_space_BO\Multi-fidelity Bayesian Optimization\Plasmid DNA production\productivity"
python BO_productivity.py
```

Example multi-fidelity DME workflow:

```powershell
cd "reduced_space_BO\Multi-fidelity Bayesian Optimization\Dimethyl ether production\energy"
python BO_energy.py
```

The DME scripts use the Aspen HYSYS model included at:

```powershell
Data generation\Dimethyl ether - Aspen HYSYS\i-dme-complete-gsa-equil.hsc
```

Generated optimization outputs are written to `results_*` folders. These are intentionally ignored by Git.

## Reproducibility Notes
The included CSV files, trained ANNs, simulator files, and notebooks are intended to support code availability and reproducibility for the published workflows. Users without SuperPro Designer or Aspen HYSYS can still inspect the data, surrogate models, and Python implementation, but they will not be able to rerun the high-fidelity simulator evaluations.

## Global Sensitivity Analysis

GSA is performed using the SobolGSA software:

https://www.imperial.ac.uk/process-systems-engineering/research/free-software/sobolgsa-software/

The relevant GSA files and generated metamodel code are included under `GSA/`.

## Citation

If you use this code in your work, please cite:

```bibtex
@article{triantafyllou2024comparative,
  title={Comparative assessment of simulation-based and surrogate-based approaches to flowsheet optimization using dimensionality reduction},
  author={Triantafyllou, Niki and Lyons, Ben and Bernardi, Andrea and Chachuat, Benoit and Kontoravdi, Cleo and Papathanasiou, Maria M.},
  journal={Computers \& Chemical Engineering},
  volume={189},
  pages={108807},
  year={2024},
  issn={0098-1354},
  publisher={Elsevier},
  doi={10.1016/j.compchemeng.2024.108807},
  url={https://www.sciencedirect.com/science/article/pii/S0098135424002254}
}
```
