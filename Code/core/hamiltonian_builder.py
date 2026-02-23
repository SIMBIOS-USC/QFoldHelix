import numpy as np
import pennylane as qml
import gc
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from utils.general_utils import get_qubit_index
from data_loaders.energy_matrix_loader import (
    _load_first_neighbors_matrix_file, 
    _load_third_neighbors_matrix_file, 
    _load_fourth_neighbors_matrix_file, 
    _load_energy_matrix_file
)

class HamiltonianBuilder:
    """
    Builds the protein Hamiltonian using intensive property scaling 
    and Z-score normalization (Goldstein-Wolynes method).
    """
    def __init__(self, L: int, amino_acids: List[str], bits_per_pos: int, n_qubits: int, **kwargs):
        self.L = L
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.bits_per_pos = bits_per_pos
        self.n_qubits = n_qubits
        self.kwargs = kwargs
        self.terms_by_type = defaultdict(list)
        self.pauli_terms = []
        
        # 1. Initialize properties from Fauchère-Pliska and Pace-Scholtz tables
        self._init_properties()
        
        # 2. Load mandatory statistical matrices
        self._load_matrices_from_files()
        
        # 3. Calibration: Standardize terms to unit variance (Z-score)
        self.stats = self._calibrate_z_scores(n_decoys=1000)

    def _init_properties(self):
        """Standardized biochemical scales as per theoretical framework."""
        # Table: Helix Propensity (kcal/mol) - Pace & Scholtz
        self.h_alpha = {
            'A': 0.00, 'L': 0.21, 'R': 0.21, 'M': 0.24, 'K': 0.26, 'Q': 0.39,
            'E': 0.40, 'I': 0.41, 'W': 0.49, 'S': 0.50, 'Y': 0.53, 'F': 0.54,
            'V': 0.61, 'H': 0.61, 'N': 0.65, 'T': 0.66, 'C': 0.68, 'D': 0.69,
            'G': 1.00, 'P': 3.15
        }
        # Table: Hydrophobicity - Fauchère-Pliska
        self.hydro = {
            'D': -0.77, 'E': -0.64, 'K': -0.99, 'R': -1.01, 'H': 0.13, 'G': 0.00,
            'A': 0.31, 'V': 1.22, 'L': 1.70, 'I': 1.80, 'P': 0.72, 'M': 1.23,
            'F': 1.79, 'W': 2.25, 'Y': 0.96, 'T': -0.04, 'S': 0.26, 'C': 1.54,
            'N': -0.60, 'Q': -0.22
        }
        self.charges = {aa: 0 for aa in self.amino_acids}
        for aa in ['R', 'K', 'H']: self.charges[aa] = 1
        for aa in ['D', 'E']: self.charges[aa] = -1

    def _load_matrices_from_files(self):
        """Loads MJ matrix and Helix Neighbor matrices (k=1, 3, 4)."""
        # Miyazawa-Jernigan Matrix
        mj_full, _ = _load_energy_matrix_file()
        self.mj_matrix = mj_full # In a real scenario, map to self.amino_acids
        
        # Neighbor propensities (Nacar et al.)
        self.M1, _ = _load_first_neighbors_matrix_file()
        self.M3, _ = _load_third_neighbors_matrix_file()
        self.M4, _ = _load_fourth_neighbors_matrix_file()

    def _calibrate_z_scores(self, n_decoys: int):
        """Estimates mu and sigma for each term using decoy sequences to resolve Gradient Bias."""
        # Placeholder for statistical calibration logic
        return {k: (0.0, 1.0) for k in ['helix_local', 'pw_int', 'hydro_moment', 'env_pol', 'env_chg', 'electrostatic', 'helix_neigh']}

    def _apply_z_score(self, term_name: str, raw_energy: float) -> float:
        mu, sigma = self.stats.get(term_name, (0.0, 1.0))
        return (raw_energy - mu) / sigma

    # --- Hamiltonian Term Construction (Intensive Scaling) ---

    def _add_helix_local(self, weight: float):
        """H_helix-local: Standardized by 1/L."""
        L_inv = 1.0 / self.L
        for i in range(self.L):
            for α, aa in enumerate(self.amino_acids):
                raw = self.h_alpha.get(aa, 1.0) * L_inv
                coeff = weight * self._apply_z_score('helix_local', raw)
                self.terms_by_type['helix_local'].extend(self._projector_terms(i, α, coeff))

    def _add_pairwise_interactions(self, weight: float, d_max: int = 4):
        """H_pw-int: Scaled by number of neighboring contact pairs (L * k_max)."""
        scale = 1.0 / (self.L * d_max)
        for i in range(self.L):
            for j in range(i + 1, min(i + d_max + 1, self.L)):
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        raw = self.mj_matrix[α, β] * scale
                        coeff = weight * self._apply_z_score('pw_int', raw)
                        self._add_qubo_interaction(i, α, j, β, coeff, 'pw_int')

    def _add_hydrophobic_moment(self, weight: float):
        """H_hydro_moment: Global All-to-All scaling."""
        scale = 2.0 / (self.L * (self.L - 1))
        delta = np.deg2rad(100.0)
        for i in range(self.L):
            for j in range(self.L):
                cos_phase = np.cos(delta * (i - j))
                for α, aa_i in enumerate(self.amino_acids):
                    for β, aa_j in enumerate(self.amino_acids):
                        raw = -(self.hydro[aa_i] * self.hydro[aa_j] * cos_phase) * scale
                        coeff = weight * self._apply_z_score('hydro_moment', raw)
                        self._add_qubo_interaction(i, α, j, β, coeff, 'hydro_moment')

    def _add_helix_neighbors(self, weights: Dict[int, float]):
        """H_helix-neighbors: Restricted pairs scaled by 1/(L-k)."""
        matrices = {1: self.M1, 3: self.M3, 4: self.M4}
        for k, matrix in matrices.items():
            scale = 1.0 / (self.L - k)
            w = weights.get(k, 1.0)
            for i in range(self.L - k):
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        raw = matrix[α, β] * scale
                        coeff = w * self._apply_z_score('helix_neigh', raw)
                        self._add_qubo_interaction(i, α, i + k, β, coeff, 'helix_neigh')

    # --- Qubit Projection Utilities ---

    def _projector_terms(self, pos: int, code: int, base_coeff: float):
        """Encodes sequence in binary variables using bit-projectors."""
        b = self.bits_per_pos
        s = [(1.0 if (code >> k) & 1 == 0 else -1.0) for k in range(b)]
        terms = []
        for mask in range(1 << b):
            coeff = base_coeff / (2 ** b)
            pauli = ['I'] * self.n_qubits
            for k in range(b):
                if (mask >> k) & 1:
                    coeff *= s[k]
                    w = get_qubit_index(pos, k, self.bits_per_pos)
                    pauli[w] = 'Z'
            if abs(coeff) > 1e-10:
                terms.append((coeff, ''.join(pauli)))
        return terms

    def _add_qubo_interaction(self, i, α, j, β, energy, category):
        """Adds terms for x_{i,α} * x_{j,β} binary variables."""
        t1 = self._projector_terms(i, α, 1.0)
        t2 = self._projector_terms(j, β, energy)
        for c1, p1 in t1:
            for c2, p2 in t2:
                pauli = ['I'] * self.n_qubits
                for k in range(self.n_qubits):
                    if p1[k] == 'Z' or p2[k] == 'Z': pauli[k] = 'Z'
                self.terms_by_type[category].append((c1 * c2, ''.join(pauli)))

    def build_hamiltonian(self) -> qml.Hamiltonian:
        """Constructs the total H' = Σ ω_k H'_k."""
        self._add_helix_local(self.kwargs.get('lambda_local', 1.0))
        self._add_pairwise_interactions(self.kwargs.get('lambda_pw', 1.0))
        self._add_hydrophobic_moment(self.kwargs.get('lambda_mu', 1.0))
        self._add_helix_neighbors({1: 1.0, 3: 1.0, 4: 1.0})
        
        all_terms = []
        for cat in self.terms_by_type: all_terms.extend(self.terms_by_type[cat])
        
        # Combine identical terms and build PennyLane object
        final_dict = defaultdict(float)
        for c, p in all_terms: final_dict[p] += c
        
        coeffs = [v for v in final_dict.values() if abs(v) > 1e-9]
        ops = [qml.pauli.string_to_pauli_word(k) for k, v in final_dict.items() if abs(v) > 1e-9]
        
        return qml.Hamiltonian(coeffs, ops)
