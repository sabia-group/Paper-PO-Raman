# Investigating anharmonicities in polarization-orientation Raman spectra of acene crystals with machine learning

**Authors:** Paolo Lazzaroni, Shubham Sharma, Mariana Rossi

**DOI:** https://doi.org/10.1103/t4s2-45t8

**Note:** This repository contains input files, configuration files, and representative outputs. Full datasets and trajectory files are omitted due to size constraints, can be obtained upon request.

## Repository Structure

### `alpha-ml/`
Machine learning models and datasets for polarizabilties on anthracene.

- **`mace/`** - MACE model files
  - `anthracene.model` - Trained MACE model for anthracene
  - `train.xyz`, `test.xyz` - Training and test datasets for model evaluation
  - `train_mace_pol.sbatch` - HPC batch submission script for model training
  - `dfpt/control.in` - FHI-aims input for DFPT calculations

### `md-ml/`
Machine learning models and datasets for MLIPs of anthracene and naphthalene.

- **`control.in-fhi-aims`** - FHI-aims input for DFT reference calculations
- **`MACE_input.sh`** - Shell script for MACE potential training
- **`n2p2_input.in`** - Configuration for N2P2 (Behler-Parrinello) neural network potential training

- **`anthra/`** - Anthracene mace model and dataset
  - `anthracene_trainset.extxyz` - Extended XYZ format training set
  - `anthra_float32_swa.model`, `anthra_float64_swa.model` - Stochastic weight averaging models at different precision

- **`naph/`** - Naphthalene mace model and dataset
  - `naphthalene_trainset.extxyz` - Training set for naphthalene
  - `naphthalene_mace_tight.model` - Tightly fitted MACE model

### `pigs/`
Temperature-elevated path-integral coarse-graining (PIGS) delta potential model and generation. Fully based on https://github.com/venkatkapil24/Te-PIGS-spectroscopy-tutorial 

- **`dataset.xyz`** - Dataset with centroid and physical forces
- **`train.sh`** - MACE training script
- **`deltaPMF_anthracene.model`** - Delta PMF model file

- **`dataset-generation/`** - PIGS dataset generation
  - `input.xml` - input configuration for i-PI
  - `nvt.centroid_force.extxyz`, `nvt.physical_force.extxyz` - NVT ensemble trajectories (centroid and physical forces)

- **`production/`** - Production PIGS runs
  - `input.xml` - i-PI input for PIGS production simulation (example usage)

### `ramanTensors/`
Phonons and Raman tensor calculations for harmonic or RGDOS spectra.

- **`phonons/`** - Phonon calculations with i-PI
  - **`minimize/`** - Geometry optimization with MACE model
    - `input.xml` - i-PI input file
    - `initial.xyz`, `optimized.xyz` - Initial and optimized structures
  
  - **`phonons/`** 
    - `input.xml` - Phonon calculation input with i-PI

- **`displacements-Rtensor.py`** - Generate +/- displaced structures along normal modes from referene geometries
- **`modes`** - Normal modes file from i-PI phonon calculation

- **`minus-displacement-example/`** - Example calculation for negative displacements with i-PI replay mode
  - `control.in`, `geometry.in` - FHI-aims DFPT setup
  - `input.xml` - i-PI replay input file
  - `init.xyz`, `minus.xyz` - Initial and displaced geometries "trajectory"
  - `pol-minus.pol_0` - Polarizability tensor for minus displaced structures

### `production-md/`
Production molecular dynamics trajectories input files.

- **`input-nve.xml`, `input-nvt.xml`** - i-PI NVE and NVT input files
- **`rgdos-example/`** - Example workflow for obtaining RGDOS correlation functions 

  - **`100K/`** 
    - `optimized.xyz` - Optimized geometry
    - `modes` - Vibrational mode file printed by i-PI phonon calculation
    - `pol-minus`, `pol-plus` - Polarizability for +/- displaced geometries
    
    - **`1x1x1/`** 
      - **`nve/run1/`** - NVE MD runs with i-PI example (1x1x1 cell)
        - `nvt.md` - MD output file
        - `nvt.chk` - NVT checkpoint file for NVE restart 
        - `nve.pos_0.xyz` - MD trajectory
        - `RESTART` - i-PI input file 
        - `anthra100K.cif` - Structure initialization for i-PI
        - `ipi.out` - i-PI simulation output
        - `run_ase.py` - Python script for ASE-based MACE model driver

---

*Last updated: January 2026*
