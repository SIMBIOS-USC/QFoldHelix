import numpy as np
import pennylane as qml
import gc
from typing import List, Tuple, Dict, Any
from collections import defaultdict

# Importaciones requeridas de tu estructura de archivos
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

    Fiel a:
      - Escalado intensivo (Imagen 3 / Sec. Thermodynamic Consistency)
      - Z-score real con N decoys aleatorios por término
      - Momento hidrofóbico con componentes x e y  (|μ_H|² = μx² + μy²)
      - Helix-neighbors sin signo negativo arbitrario
    """

    def __init__(
        self,
        L: int,
        amino_acids: List[str],
        bits_per_pos: int,
        n_qubits: int,
        **kwargs
    ):
        self.L = L
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.bits_per_pos = bits_per_pos
        self.n_qubits = n_qubits
        self.kwargs = kwargs
        self.terms_by_type = defaultdict(list)
        self.pauli_terms = []

        # 1. Propiedades físico-químicas (Tablas 1 y 2 del paper)
        self._init_properties()

        # 2. Matrices MJ y vecinos estadísticos k=1,3,4
        self._load_matrices_from_files()

        # 3. Calibración Z-score real (Goldstein et al. 1992)
        #    Evalúa n_decoys secuencias aleatorias para estimar μ y σ de cada término
        n_decoys = self.kwargs.get('n_decoys', 10000)
        self.stats = self._calibrate_z_scores(n_decoys=n_decoys)

    # ------------------------------------------------------------------
    # Inicialización de propiedades
    # ------------------------------------------------------------------

    def _init_properties(self):
        """
        Pace-Scholtz (hélice) y Fauchère-Pliska (hidrofobicidad).
        Tabla 1 y Tabla 2 del paper.
        """
        # Propensión a hélice (kcal/mol) — menor valor = mayor propensión
        self.h_alpha = {
            'A': 0.00, 'L': 0.21, 'R': 0.21, 'M': 0.24, 'K': 0.26, 'Q': 0.39,
            'E': 0.40, 'I': 0.41, 'W': 0.49, 'S': 0.50, 'Y': 0.53, 'F': 0.54,
            'V': 0.61, 'H': 0.61, 'N': 0.65, 'T': 0.66, 'C': 0.68, 'D': 0.69,
            'G': 1.00, 'P': 3.15
        }
        # Hidrofobicidad Fauchère-Pliska: >0 Apolar, <0 Polar
        self.hydro = {
            'D': -0.77, 'E': -0.64, 'K': -0.99, 'R': -1.01, 'H':  0.13,
            'G':  0.00, 'A':  0.31, 'V':  1.22, 'L':  1.70, 'I':  1.80,
            'P':  0.72, 'M':  1.23, 'F':  1.79, 'W':  2.25, 'Y':  0.96,
            'T': -0.04, 'S':  0.26, 'C':  1.54, 'N': -0.60, 'Q': -0.22
        }
        # Cargas formales
        self.charges = {aa: 0 for aa in self.amino_acids}
        for aa in ['R', 'K', 'H']:
            self.charges[aa] = 1
        for aa in ['D', 'E']:
            self.charges[aa] = -1

    def _load_matrices_from_files(self):
        """Carga MJ y matrices de vecinos estadísticos k=1,3,4."""
        mj_full, mj_symbols = _load_energy_matrix_file()
        aa_to_idx = {aa: i for i, aa in enumerate(mj_symbols)}
        self.mj_matrix = np.zeros((self.n_aa, self.n_aa))
        for i, aa1 in enumerate(self.amino_acids):
            for j, aa2 in enumerate(self.amino_acids):
                self.mj_matrix[i, j] = mj_full[aa_to_idx[aa1], aa_to_idx[aa2]]

        self.M1, _ = _load_first_neighbors_matrix_file()
        self.M3, _ = _load_third_neighbors_matrix_file()
        self.M4, _ = _load_fourth_neighbors_matrix_file()

    # ------------------------------------------------------------------
    # Entorno (Helical Wheel)
    # ------------------------------------------------------------------

    def _get_env(self, i: int) -> str:
        phi0 = self.kwargs.get('wheel_phase_deg', 0.0)
        # Extraer el ancho de la cara de la membrana (por defecto 90 si no viene)
        half_width = self.kwargs.get('wheel_halfwidth_deg', 90.0)
        
        # Ángulo del residuo i (100° por residuo en hélice alfa)
        theta = ((i - 1) * 100.0 + phi0) % 360.0
        
        # Normalizar a rango [-180, 180]
        angle = theta if theta <= 180.0 else theta - 360.0
        
        return "membrane" if abs(angle) <= half_width else "water"

    # ------------------------------------------------------------------
    # Z-score real  (Sec. 2 del paper: Goldstein 1992)
    # ------------------------------------------------------------------

    def _raw_helix_local(self, seq: List[int]) -> float:
        scale = 1.0 / self.L
        return sum(self.h_alpha.get(self.amino_acids[a], 1.0) for a in seq) * scale

    def _raw_env_pol(self, seq: List[int]) -> float:
        """
        Actua en AMBOS entornos (simetria anfipática):
          Membrana: -h -> premia apolares (h>0), penaliza polares (h<0)
          Agua:     +h -> premia polares  (h<0), penaliza apolares (h>0)
        """
        scale = 1.0 / self.L
        total = 0.0
        for i, a in enumerate(seq, start=1):
            h = self.hydro[self.amino_acids[a]]
            total += -h if self._get_env(i) == "membrane" else +h
        return total * scale

    def _raw_env_chg(self, seq: List[int]) -> float:
        """
        Actua SOLO en agua. Premio UNIDIRECCIONAL: solo favorece la carga
        OPUESTA a sigma, nunca penaliza la carga del mismo signo.
          neg (sigma=-1): K,R,H en agua -> min((-1)(+1), 0) = -1  FAV
                          D,E   en agua -> min((-1)(-1), 0) =  0  NEUTRO
          pos (sigma=+1): D,E   en agua -> min((+1)(-1), 0) = -1  FAV
                          K,R,H en agua -> min((+1)(+1), 0) =  0  NEUTRO
        Residuos neutros (q=0): sin efecto en cualquier caso.
        """
        scale = 1.0 / self.L
        m_charge = self.kwargs.get('membrane_charge', 'neg').lower()
        sigma = {'neg': -1.0, 'pos': 1.0, 'neu': 0.0}.get(m_charge, 0.0)
        total = 0.0
        for i, a in enumerate(seq, start=1):
            if self._get_env(i) != "membrane":   # solo en agua
                raw = sigma * self.charges[self.amino_acids[a]]
                total += min(raw, 0.0)            # solo el componente favorable
        return total * scale

    def _raw_pw_int(self, seq: List[int]) -> float:
        d_max = self.kwargs.get('max_interaction_dist', 1)
        scale = 1.0 / (self.L * d_max)
        total = 0.0
        for i in range(self.L):
            for j in range(i + 1, min(i + d_max + 1, self.L)):
                total += self.mj_matrix[seq[i], seq[j]]
        return total * scale

    def _raw_hydro_moment(self, seq: List[int]) -> float:
        """
        |μ_H|² = (Σ H_α cos(δi) x_{i,α})² + (Σ H_α sin(δi) x_{i,α})²
        Escalado global 2/(L(L-1)).
        """
        scale = 2.0 / (self.L * (self.L - 1)) if self.L > 1 else 1.0
        delta = np.deg2rad(100.0)
        mu_x = sum(self.hydro[self.amino_acids[seq[i]]] * np.cos(delta * (i + 1))
                   for i in range(self.L))
        mu_y = sum(self.hydro[self.amino_acids[seq[i]]] * np.sin(delta * (i + 1))
                   for i in range(self.L))
        return -(mu_x ** 2 + mu_y ** 2) * scale

    def _raw_electrostatic(self, seq: List[int]) -> float:
        r_c = self.kwargs.get('electrostatic_cutoff', 8)
        scale = 2.0 / (self.L * (self.L - 1)) if self.L > 1 else 1.0
        total = 0.0
        for i in range(self.L):
            env_i = self._get_env(i + 1)
            for j in range(i + 1, min(self.L, i + r_c + 1)):
                env_j = self._get_env(j + 1)
                if env_i == "membrane" and env_j == "membrane":
                    eps = 0.1
                elif env_i == "water" and env_j == "water":
                    eps = 1.0
                else:
                    eps = 0.3
                dist = j - i
                qi = self.charges[self.amino_acids[seq[i]]]
                qj = self.charges[self.amino_acids[seq[j]]]
                total += (qi * qj) / ((1 + dist) * eps)
        return total * scale

    def _raw_helix_neigh(self, seq: List[int]) -> float:
        matrices = {1: self.M1, 3: self.M3, 4: self.M4}
        total = 0.0
        for k, matrix in matrices.items():
            if self.L <= k:
                continue
            scale = 1.0 / (self.L - k)
            for i in range(self.L - k):
                total += matrix[seq[i], seq[i + k]] * scale
        return total

    def _calibrate_z_scores(self, n_decoys: int) -> Dict[str, Tuple[float, float]]:
        """
        Estima μ y σ para cada término evaluando n_decoys secuencias aleatorias.
        Implementación fiel al paper (Goldstein et al. 1992, Sec. 2.2).
        """
        rng = np.random.default_rng(self.kwargs.get('zscore_seed', 42))

        accum = {
            'helix_local':   [],
            'env_pol':       [],
            'env_chg':       [],
            'pw_int':        [],
            'hydro_moment':  [],
            'electrostatic': [],
            'helix_neigh':   [],
        }

        for _ in range(n_decoys):
            seq = rng.integers(0, self.n_aa, size=self.L).tolist()
            accum['helix_local'].append(self._raw_helix_local(seq))
            accum['env_pol'].append(self._raw_env_pol(seq))
            accum['env_chg'].append(self._raw_env_chg(seq))
            accum['pw_int'].append(self._raw_pw_int(seq))
            accum['hydro_moment'].append(self._raw_hydro_moment(seq))
            accum['electrostatic'].append(self._raw_electrostatic(seq))
            accum['helix_neigh'].append(self._raw_helix_neigh(seq))

        stats = {}
        for key, values in accum.items():
            arr = np.array(values)
            mu = float(np.mean(arr))
            sigma = float(np.std(arr))
            # Evitar división por cero: si σ≈0 el término es constante → no aporta gradiente
            stats[key] = (mu, sigma if sigma > 1e-12 else 1.0)

        return stats

    def _apply_z_score(self, term_name: str, raw_energy: float) -> float:
        """H' = (H - μ) / σ  (Ec. Z-score del paper)."""
        mu, sigma = self.stats.get(term_name, (0.0, 1.0))
        return (raw_energy - mu) / sigma

    # ------------------------------------------------------------------
    # Términos Hamiltonianos con escalado intensivo
    # ------------------------------------------------------------------

    def _add_helix_local(self, weight: float):
        """H_helix-local (1-body): Escalado 1/L."""
        scale = 1.0 / self.L
        for i in range(1, self.L + 1):
            for alpha, aa in enumerate(self.amino_acids):
                raw = self.h_alpha.get(aa, 1.0) * scale
                coeff = weight * self._apply_z_score('helix_local', raw)
                self.terms_by_type['helix_local'].extend(
                    self._projector_terms(i - 1, alpha, coeff)
                )

    def _add_env_mediated(self, w_pol: float, w_chg: float):
        """
        H_env (1-body): Partición hidrofóbica + acoplamiento de carga superficial.
        Escalado 1/L. Premio negativo para estabilidad biológica.

        Convención de carga (paper Ec. H_env-chg = λ · σ · q_α · 𝟙_mem):
          membrane_charge='neg'  →  σ = -1  (membrana bacteriana, aniónica)
              σ · q = (-1)(+1) = -1  →  K,R,H favorecidos  ✓
              σ · q = (-1)(-1) = +1  →  D,E  penalizados    ✓
          membrane_charge='pos'  →  σ = +1  (membrana catiónica)
              σ · q = (+1)(-1) = -1  →  D,E  favorecidos    ✓
          membrane_charge='neu'  →  σ =  0  (sin acoplamiento electrostático)

        La energía NEGATIVA = favorable = estabiliza la secuencia.
        No se invierte el signo: σ·q ya produce el comportamiento correcto.
        """
        scale = 1.0 / self.L
        m_charge = self.kwargs.get('membrane_charge', 'neg').lower()
        # σ: potencial de superficie de la membrana
        sigma = {'neg': -1.0, 'pos': 1.0, 'neu': 0.0}.get(m_charge, 0.0)

        for i in range(1, self.L + 1):
            env = self._get_env(i)
            for alpha, aa in enumerate(self.amino_acids):
                h = self.hydro[aa]
                q = self.charges[aa]
                # env_pol: AMBOS entornos (simetria anfipática).
                #   Membrana: -h -> premia apolares (h>0), penaliza polares (h<0)
                #   Agua:     +h -> premia polares  (h<0), penaliza apolares (h>0)
                raw_pol = (h if env == "membrane" else -h) * scale
                # env_chg: SOLO agua. Premio unidireccional: solo carga OPUESTA a sigma.
                #   neg: K,R,H en agua -> min((-1)(+1),0) = -1 FAV | D,E -> 0 NEUTRO
                #   pos: D,E   en agua -> min((+1)(-1),0) = -1 FAV | K,R -> 0 NEUTRO
                raw_chg = (min(sigma * q, 0.0) if env != "membrane" else 0.0) * scale
                c = (
                    w_pol * self._apply_z_score('env_pol', raw_pol)
                    + w_chg * self._apply_z_score('env_chg', raw_chg)
                )
                self.terms_by_type['env'].extend(
                    self._projector_terms(i - 1, alpha, c)
                )

    def _add_pairwise_interactions(self, weight: float, d_max: int = 1):
        """H_pw-int: MJ dentro de ventana d_max. Escalado 1/(L·d_max)."""
        scale = 1.0 / (self.L * d_max)
        for i in range(self.L):
            for j in range(i + 1, min(i + d_max + 1, self.L)):
                for alpha in range(self.n_aa):
                    for beta in range(self.n_aa):
                        raw = self.mj_matrix[alpha, beta] * scale
                        coeff = weight * self._apply_z_score('pw_int', raw)
                        self._add_qubo_interaction(i, alpha, j, beta, coeff, 'pw_int')

    def _add_hydrophobic_moment(self, weight: float):
        """
        H_hydro-moment (All-to-All): |μ_H|² con componentes x e y.
        Escalado 2/(L(L-1)).

        CORRECCIÓN respecto a la versión anterior:
          Se incluye el término seno además del coseno, fiel a la Ec. del paper:
          H_μ = -λ Σ_{i,α} Σ_{j,β} (a_{i,α}·a_{j,β} + b_{i,α}·b_{j,β}) x_{i,α} x_{j,β}
          donde a_{i,α} = H_α cos(δi),  b_{i,α} = H_α sin(δi).
        """
        scale = 2.0 / (self.L * (self.L - 1)) if self.L > 1 else 1.0
        delta = np.deg2rad(100.0)

        for i in range(1, self.L + 1):
            for j in range(i + 1, self.L + 1):
                # Factores angulares
                cos_ij = np.cos(delta * (i - j))  # cos(δi)cos(δj) + sin(δi)sin(δj) = cos(δ(i-j))
                for alpha, aa_i in enumerate(self.amino_acids):
                    for beta, aa_j in enumerate(self.amino_acids):
                        # a·a + b·b = H_α H_β (cos(δi)cos(δj) + sin(δi)sin(δj))
                        #           = H_α H_β cos(δ(i-j))
                        raw = -(self.hydro[aa_i] * self.hydro[aa_j] * cos_ij) * scale
                        coeff = weight * self._apply_z_score('hydro_moment', raw)
                        self._add_qubo_interaction(
                            i - 1, alpha, j - 1, beta, coeff, 'hydro_moment'
                        )

    def _add_electrostatic_screening(self, weight: float, r_c: int = 8):
        """
        H_electrostatic: Dieléctrico dependiente del entorno.
        Escalado global 2/(L(L-1)).
        """
        scale = 2.0 / (self.L * (self.L - 1)) if self.L > 1 else 1.0
        for i in range(1, self.L + 1):
            env_i = self._get_env(i)
            for j in range(i + 1, min(self.L + 1, i + r_c + 1)):
                env_j = self._get_env(j)
                if env_i == "membrane" and env_j == "membrane":
                    eps = 0.1
                elif env_i == "water" and env_j == "water":
                    eps = 1.0
                else:
                    eps = 0.3
                dist = abs(i - j)
                for alpha, aa_i in enumerate(self.amino_acids):
                    for beta, aa_j in enumerate(self.amino_acids):
                        raw = (
                            self.charges[aa_i] * self.charges[aa_j]
                            / ((1 + dist) * eps)
                        ) * scale
                        coeff = weight * self._apply_z_score('electrostatic', raw)
                        self._add_qubo_interaction(
                            i - 1, alpha, j - 1, beta, coeff, 'electrostatic'
                        )

    def _add_helix_neighbors(self, weight: float):
        """
        H_helix-neighbors: Matrices estadísticas k=1,3,4 (Nacar et al.).
        Escalado 1/(L-k).

        CORRECCIÓN: Se elimina el signo negativo arbitrario de la versión anterior.
        El paper define directamente +M^(k)_{αβ}, sin inversión de signo.
        """
        matrices = {1: self.M1, 3: self.M3, 4: self.M4}
        for k, matrix in matrices.items():
            if self.L <= k:
                continue
            scale = 1.0 / (self.L - k)
            for i in range(self.L - k):
                for alpha in range(self.n_aa):
                    for beta in range(self.n_aa):
                        raw = matrix[alpha, beta] * scale  # sin signo negativo
                        coeff = weight * self._apply_z_score('helix_neigh', raw)
                        self._add_qubo_interaction(
                            i, alpha, i + k, beta, coeff, 'helix_neigh'
                        )

    # ------------------------------------------------------------------
    # Utilidades QUBO
    # ------------------------------------------------------------------

    def _projector_terms(self, pos: int, code: int, base_coeff: float):
        """
        Expansión del proyector one-hot en base Pauli-Z.
        x_{i,α} = Π_k (1 ± Z_k) / 2
        """
        b = self.bits_per_pos
        s = [(1.0 if (code >> (b-1-k)) & 1 == 0 else -1.0) for k in range(b)]
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

    def _add_qubo_interaction(self, i, alpha, j, beta, energy, category):
        """Producto tensorial de dos proyectores → término de 2 cuerpos."""
        t1 = self._projector_terms(i, alpha, 1.0)
        t2 = self._projector_terms(j, beta, energy)
        for c1, p1 in t1:
            for c2, p2 in t2:
                pauli = ['I'] * self.n_qubits
                for k in range(self.n_qubits):
                    if p1[k] == 'Z' or p2[k] == 'Z':
                        pauli[k] = 'Z'
                self.terms_by_type[category].append((c1 * c2, ''.join(pauli)))

    # ------------------------------------------------------------------
    # Ensamblado del Hamiltoniano
    # ------------------------------------------------------------------

    def build_hamiltonian(self, backend: str = 'pennylane') -> Tuple[List, Any]:
        """
        Ensambla el Hamiltoniano total ponderado:
          H_total = Σ_k ω_k H'_k
        """
        self._add_helix_local(
            self.kwargs.get('lambda_local', 1.0)
        )
        self._add_env_mediated(
            self.kwargs.get('lambda_env', 1.0),
            self.kwargs.get('lambda_charge', 1.0)
        )
        self._add_pairwise_interactions(
            self.kwargs.get('lambda_pairwise', 1.0),
            self.kwargs.get('max_interaction_dist', 1)
        )
        self._add_hydrophobic_moment(
            self.kwargs.get('lambda_mu', 1.0)
        )
        self._add_electrostatic_screening(
            self.kwargs.get('lambda_electrostatic', 1.0),
            self.kwargs.get('electrostatic_cutoff', 8)
        )
        self._add_helix_neighbors(
            self.kwargs.get('lambda_helix_pairs', 1.0)
        )

        # Consolidar términos Pauli (sumar coeficientes de strings idénticos)
        final_dict = defaultdict(float)
        for cat in self.terms_by_type:
            for c, p in self.terms_by_type[cat]:
                final_dict[p] += c

        pauli_list = [
            (c, p) for p, c in final_dict.items() if abs(c) > 1e-9
        ]
        coeffs = [t[0] for t in pauli_list]
        ops = [qml.pauli.string_to_pauli_word(t[1]) for t in pauli_list]

        return pauli_list, qml.Hamiltonian(coeffs, ops)
