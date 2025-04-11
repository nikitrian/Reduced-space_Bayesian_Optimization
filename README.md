# Comparative assessment of simulation-based and surrogate-based approaches to flowsheet optimization using dimensionality reduction 

Niki Triantafyllou, Ben Lyons, Andrea Bernardi, Benoit Chachuat, Cleo Kontoravdi, and Maria M. Papathanasiou

This is the official implementation of our *Computers & Chemical Engineering* 2024 [paper](https://www.sciencedirect.com/science/article/pii/S0098135424002254).

## Algorithms
This repository contains the reduced-space simulation-based and surrogate-based Bayesian optimization algorithms for the optimization of simulator-based process flowsheets.  
The framework employs global sensitivity analysis (GSA) for dimensionality reduction by identifying critical process variables that contribute significantly to the variability of the objective function (e.g., productivity and operating costs).  

The algorithms are demonstrated on two case studies:
- A biopharmaceutical process for the production of plasmid DNA  
- A chemical process for the production of dimethyl ether (DME)

We also provide code for exact optimization of the ANN surrogates using the [OMLT](https://github.com/cog-imperial/OMLT) package.

## Dataset

This repository includes:
- Python–Aspen HYSYS connection scripts
- Python–VBA–SuperPro Designer interface code for data generation

## Global Sensitivity Analysis (GSA)

GSA is performed using the [SobolGSA software](https://www.imperial.ac.uk/process-systems-engineering/research/free-software/sobolgsa-software/).  
The relevant code for GSA is also included.

## Citation
If you use this code in your work, please cite our paper:

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
