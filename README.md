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
cd QFoldHelix/environments
conda env create -f environment_cpu.yml
conda activate protein_seq_quantum_CPU
```





# ⚙️ Command-Line Arguments

The `main_final.py` script is fully configurable through command-line arguments.  
Below is a detailed description of all available parameters, grouped by functionality.

---

## 🔹 1. Main Configuration & Solver

| Flag | Default | Description |
|------|----------|------------|
| `-L`, `--length` | `4` | Total length of the amino acid sequence to design. |
| `-R`, `--residues` | `"V,Q,L,R"` | Allowed amino acids (search space), comma-separated. |
| `-b`, `--backend` | `"pennylane"` | Quantum backend. Options: `pennylane`, `qiskit`. |
| `--solver` | `"qaoa"` | Optimization method: `qaoa`, `vqe`, `classical`, `annealing`. |
| `--shots` | `10000000` | Number of measurement shots for quantum evaluation. |
| `--output_dir` | `"output"` | Directory where output files are saved. |

---

## 🔹 2. Membrane & Environment Definition

### Membrane Definition Modes

| Flag | Default | Description |
|------|----------|------------|
| `--membrane_mode` | `"wheel"` | Strategy: `span`, `set`, or `wheel`. |
| `--membrane` | `None` | Continuous membrane span (e.g., `1:4`). |
| `--membrane_positions` | `None` | Explicit membrane indices (e.g., `0,2,5`). |

### Helical Wheel Parameters

| Flag | Default | Description |
|------|----------|------------|
| `--wheel_phase_deg` | `-50.0` | Initial phase angle (degrees). |
| `--wheel_halfwidth_deg` | `120.0` | Half-width of membrane sector (degrees). |
| `--membrane_charge` | `"pos"` | Membrane charge: `neu`, `neg`, `pos`. |

---

## 🔹 3. Hamiltonian Weights (λ)

| Flag | Default | Description |
|---------|----------|------------|
| `--lambda_env` | `-3.0` | Environment preference (hydrophobic/polar partition). |
| `--lambda_charge` | `3.0` | Membrane surface electrostatic interaction. |
| `--lambda_mu` | `0.2` | Hydrophobic moment (amphipathicity). |
| `--lambda_local` | `1.0` | Intrinsic alpha-helix propensity. |
| `--lambda_pairwise` | `0.1` | Empirical pairwise interactions. |
| `--lambda_helix_pairs` | `1.0` | Helix-stabilizing correlations (i,i+1 / i+3 / i+4). |
| `--lambda_electrostatic` | `0.5` | General electrostatic screening term. |
| `--max_interaction_dist` | `1` | Max sequence distance for pairwise interactions. |

---

# 🚀 Example Usage



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


# 🧬 Quantum-Classical De Novo Design of Membrane-Interfacing α-Helical Peptides

This repository implements a hybrid quantum-classical framework for the **de novo design of α-helical peptides** engineered to bind protein targets at the membrane–water interface.

Designing sequences that stably fold into α-helices while maintaining functional orientation parallel to lipid bilayers remains a central challenge in structural biology. While deep learning methods (e.g., AlphaFold2) excel at structure prediction, they rely heavily on multiple sequence alignments and homology sets, limiting their applicability in *de novo* design or heterogeneous membrane environments. Conversely, traditional physics-based *ab initio* approaches face a prohibitive combinatorial explosion in sequence–conformation space.

This framework bridges that gap by encoding physically grounded constraints into a **Quadratic Unconstrained Binary Optimization (QUBO)** formulation, solvable via quantum algorithms such as QAOA or quantum annealing emulators.

---

## ✨ Key Features & Methodology

### 🔬 Physics-Based Hamiltonian

The model explicitly incorporates biophysical interactions critical for membrane stability and helix formation:

- **Intrinsic helix-forming propensities** *(H_helix-local)*  
- **Empirical pairwise interactions** (Miyazawa–Jernigan matrix) *(H_pairwise-interactions)*  
- **Amphiphilicity via hydrophobic moment optimization** *(H_hydrophobic-moment)*  
- **Membrane-environment mediation** (solvation polarity and surface charge coupling) *(H_env)*  
- **Distance-dependent electrostatic screening** *(H_electrostatic)*  
- **Statistical helix neighbor cooperativity (k = 1, 3, 4)** *(H_helix-neighbors)*  

---

### 📊 Thermodynamic Scaling & Z-Score Normalization

To prevent numerical dominance of large energy terms (gradient bias) and preserve intensive thermodynamic behavior across peptide lengths, all Hamiltonian components are rigorously scaled and transformed into standardized **Z-score space** using random decoy sequence ensembles.

---

### 🧮 Logarithmic Binary Encoding

Instead of conventional one-hot encoding requiring **O(L|A|)** qubits, this project employs a compact binary encoding scheme requiring only:

O(L log₂ |A|)

This dramatically reduces quantum resource overhead and enables larger peptide designs on near-term quantum hardware.

---

### 🧠 Amino Acid Dimensionality Reduction

Principal Component Analysis (PCA) combined with pairwise physicochemical distance matrices is used to cluster amino acids into rationalized reduced alphabets. This reduces the effective search space while preserving relevant biochemical diversity.

---

## 💻 Available Solvers

The QUBO formulation is hardware-agnostic and supports multiple resolution paradigms:

### 🔹 QAOA (Quantum Approximate Optimization Algorithm)

A hybrid variational quantum-classical algorithm implemented via Qiskit or PennyLane.  
Features:

- Superposition-based exploration of sequence space  
- CVaR (Conditional Value at Risk) objective evaluation  
- COBYLA-based classical parameter optimization  

---

### 🔹 Quantum Annealing Emulator (dwave-neal)

A high-performance classical simulated annealing backend serving as a scalable heuristic baseline for larger peptide systems.

---

### 🔹 Classical Methods

- **Exact brute-force search** (small systems) for guaranteed global minima  
- **Massive random sampling + hill climbing** heuristic for intractable sizes  

---

## 🚀 Vision

This framework demonstrates how **quantum optimization can be systematically integrated with rigorous physical modeling** to tackle combinatorial molecular design challenges, providing a scalable pathway toward membrane-aware peptide engineering on emerging quantum hardware.


