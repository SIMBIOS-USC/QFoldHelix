# Quantum Sequence Helix

🧬 **Quantum Protein Design with QAOA** 🧬  
This repository implements a **quantum approach to protein sequence design**, formulating it as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem.  
It includes QAOA/VQE (Qiskit), classical search, and simulated annealing workflows for peptide and helix sequence optimization.

---

## 🚀 Features

- **Quantum Protein Design**  
  - Encodes amino acid sequences into binary strings with `log2(N)` qubits per position  
  - Builds Hamiltonians including:
    - Local amino acid preferences  
    - Pairwise interaction terms (Miyazawa–Jernigan)  
    - Helix pair propensity terms  
    - Hydrophobic moment and environment contributions  
    - Membrane interaction terms  

- **Backends**  
  - [Qiskit](https://qiskit.org) for `qaoa` and `vqe`
  - PennyLane is used for Hamiltonian construction utilities

- **Optimization**  
  - QAOA with warm-starts and layered parameter initialization  
  - Classical brute-force solver for validation  
  - Simulated annealing solver (`neal`)  
  - Convergence tracking and energy analysis

- **Visualization**  
  - Optimization convergence plots  
  - Alpha-helix wheel diagrams with membrane/water partition  

---

## 📦 Installation

Clone the repository and create the environment using **conda**:

```bash
git clone https://github.com/AngelPineiro-USC/QFoldHelix.git
cd QFoldHelix
conda env create -f environment.yml
conda activate quantum_protein_design
```

## Usage

```bash
python Code/main_final.py -L 6 -R V,Q,N,S \
    --backend qiskit \
    --solver qaoa \
    --lambda_pairwise 0.5 \
    --lambda_helix_pairs 0.5 \
    --lambda_env 5.0 \
    --lambda_local 0.2 \
    --membrane_mode wheel \
    --wheel_phase_deg 0 \
    --wheel_halfwidth_deg 90
```

Quick alternatives:

```bash
# Classical brute force
python Code/main_final.py --solver classical -L 4 -R V,Q,L,R

# Simulated annealing
python Code/main_final.py --solver annealing -L 6 -R V,Q,N,S
```
