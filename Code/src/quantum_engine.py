import os
import numpy as np
import time
import math
from typing import Dict, List, Any
import random

# Forzar un solo hilo para evitar bloqueos
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["QISKIT_IN_PARALLEL"] = "FALSE"

# Imports Qiskit
QISKIT_AVAILABLE = True
QISKIT_IMPORT_ERROR = None
try:
    from qiskit import transpile
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit_aer import AerSimulator
    from qiskit_algorithms.optimizers import COBYLA
except ImportError as e:
    QISKIT_AVAILABLE = False
    QISKIT_IMPORT_ERROR = e
    transpile = None
    SparsePauliOp = None
    QAOAAnsatz = None
    AerSimulator = None
    COBYLA = None

# Import IBM (Opcional)
if QISKIT_AVAILABLE:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        IBM_AVAILABLE = True
    except ImportError:
        IBM_AVAILABLE = False
else:
    IBM_AVAILABLE = False

from .utils_logic import decode_solution_logic
from visualization.plot_utils import ProteinPlotter
from core.hamiltonian_builder import HamiltonianBuilder

class QuantumProteinDesign:
    def __init__(self, sequence_length: int, amino_acids: List[str], **kwargs):
        self.L = sequence_length
        self.amino_acids = amino_acids
        self.n_aa = len(amino_acids)
        self.bits_per_pos = max(1, int(np.ceil(np.log2(self.n_aa))))
        self.n_qubits = self.L * self.bits_per_pos
        self.backend_name = kwargs.get('quantum_backend', 'pennylane')
        self.shots = kwargs.get('shots', 5000)
        self.output_dir = kwargs.get('output_dir', 'output')
        
        self.plotter = ProteinPlotter(output_dir=self.output_dir)
        
        self.hamiltonian_builder = HamiltonianBuilder(
            L=self.L, amino_acids=self.amino_acids, 
            bits_per_pos=self.bits_per_pos, n_qubits=self.n_qubits, **kwargs
        )
        self.pauli_terms, self.cost_hamiltonian = self.hamiltonian_builder.build_hamiltonian(self.backend_name)
        self._sanitize_hamiltonian()

    def _sanitize_hamiltonian(self):
        self.pauli_terms = [(float(c.real), p) for c, p in self.pauli_terms]
        if QISKIT_AVAILABLE and SparsePauliOp is not None:
            self.qiskit_hamiltonian = SparsePauliOp.from_list([(p, float(c)) for c, p in self.pauli_terms])
        else:
            self.qiskit_hamiltonian = None

    def _ensure_qiskit(self, feature_name: str):
        if not QISKIT_AVAILABLE:
            raise ImportError(
                f"{feature_name} requiere Qiskit y paquetes asociados. "
                "Instala: qiskit qiskit-aer qiskit-algorithms"
            ) from QISKIT_IMPORT_ERROR

    def decode_solution(self, bitstring: str) -> str:
        if not bitstring: return 'X' * self.L
        return decode_solution_logic(bitstring, self.L, self.bits_per_pos, self.amino_acids)

    def compute_energy_from_bitstring(self, bitstring: str) -> float:
        if not bitstring: return float('inf')
        z_vals = np.array([1 if b == '0' else -1 for b in bitstring])
        energy = 0.0
        for coeff, pauli_string in self.pauli_terms:
            prod = 1.0
            for i, p in enumerate(pauli_string):
                if p == 'Z': prod *= z_vals[i]
            energy += coeff * prod
        return float(energy)

    def compute_energy_breakdown(self, bitstring: str) -> Dict[str, float]:
        if not hasattr(self.hamiltonian_builder, 'terms_by_type'):
            return {'Total': self.compute_energy_from_bitstring(bitstring)}
        
        z_vals = np.array([1 if b == '0' else -1 for b in bitstring])
        breakdown = {}
        for category, terms in self.hamiltonian_builder.terms_by_type.items():
            cat_energy = 0.0
            for coeff, pauli_string in terms:
                prod = 1.0
                for i, p in enumerate(pauli_string):
                    if p == 'Z': prod *= z_vals[i]
                cat_energy += float(coeff.real) * prod
            breakdown[category] = cat_energy
        breakdown['Total'] = sum(breakdown.values())
        return breakdown

    def _get_backend(self):
        self._ensure_qiskit("El backend cuántico")
        sim_options = {"method": "statevector"} 
        return AerSimulator(device='CPU', **sim_options)

    # ------------------------------------------------------------------
    #  SOLVER CLÁSICO
    # ------------------------------------------------------------------
    def solve_classical_brute_force(self):
        print(f"\n⚡ INICIANDO SOLVER CLÁSICO (Búsqueda de Mínima Energía)")
        print(f"   Espacio de búsqueda: 2^{self.n_qubits} estados posibles.")
        
        visited_solutions = {} 
        
        if self.n_qubits <= 20:
            total_states = 2**self.n_qubits
            print(f"   🔍 Calculando la energía de TODAS las {total_states} secuencias...")
            for i in range(total_states):
                bs = format(i, f'0{self.n_qubits}b')
                en = self.compute_energy_from_bitstring(bs)
                visited_solutions[bs] = en
                if total_states > 50000 and i % (total_states//5) == 0:
                    print(f"      ... {i}/{total_states} procesados")
        else:
            print(f"   ⚠️ Espacio gigante. Usando Muestreo Inteligente.")
            n_samples = 300000 
            for _ in range(n_samples):
                bs_arr = np.random.randint(0, 2, self.n_qubits)
                bs = "".join(map(str, bs_arr))
                visited_solutions[bs] = self.compute_energy_from_bitstring(bs)
            
            best_seeds = sorted(visited_solutions.items(), key=lambda x: x[1])[:50]
            print("      ⛰️  Refinando los mejores candidatos...")
            for start_bs, start_en in best_seeds:
                curr_bs_arr = np.array([int(b) for b in start_bs])
                curr_en = start_en
                improved = True
                while improved:
                    improved = False
                    for bit_idx in range(self.n_qubits):
                        next_bs_arr = curr_bs_arr.copy()
                        next_bs_arr[bit_idx] = 1 - next_bs_arr[bit_idx]
                        next_bs = "".join(map(str, next_bs_arr))
                        if next_bs not in visited_solutions:
                            next_en = self.compute_energy_from_bitstring(next_bs)
                            visited_solutions[next_bs] = next_en
                        else: next_en = visited_solutions[next_bs]
                        if next_en < curr_en:
                            curr_en = next_en
                            curr_bs_arr = next_bs_arr
                            improved = True
                            break

        sorted_sol = sorted(visited_solutions.items(), key=lambda x: x[1])
        top_k = min(len(sorted_sol), 100)
        top_candidates = sorted_sol[:top_k]
        
        print("\n" + "="*60)
        print("🏆 TOP 10 SECUENCIAS (MENOR ENERGÍA)")
        print("="*60)
        print(f"{'Rank':<5} | {'Secuencia':<15} | {'Energía':<12}")
        print("-" * 60)
        for rank, (bs, en) in enumerate(top_candidates[:10], 1):
            print(f"{rank:<5} | {self.decode_solution(bs):<15} | {en:.6f}")
        print("-" * 60 + "\n")

        energies = np.array([en for _, en in top_candidates])
        bitstrings = [bs for bs, _ in top_candidates]
        std_dev = np.std(energies)
        T_viz = max(std_dev, 0.1) * 2.0 
        min_e = np.min(energies)
        weights = np.exp(-(energies - min_e) / T_viz)
        probs = weights / np.sum(weights)
        probs_dict = {bs: p for bs, p in zip(bitstrings, probs)}

        return {
            'bitstring': top_candidates[0][0],
            'energy': top_candidates[0][1],
            'repaired_sequence': self.decode_solution(top_candidates[0][0]),
            'repaired_cost': top_candidates[0][1],
            'probs': probs_dict,
            'costs': [] 
        }

    # ------------------------------------------------------------------
    #  SOLVER QAOA (MODO CVaR ACTIVADO)
    # ------------------------------------------------------------------
    def solve_qaoa_qiskit(self, p_layers=1, max_iter=300, ibm_token=None):
        self._ensure_qiskit("El solver QAOA")
        if self.qiskit_hamiltonian is None:
            raise RuntimeError("Hamiltoniano Qiskit no disponible tras la inicialización.")
        backend_local = self._get_backend()
        
        # --- CONFIGURACIÓN ---
        if self.n_qubits >= 36: # L=10
            p_auto, n_restarts, opt_shots, tol = 1, 1, 300, 1e-9
            use_cvar = False
        elif self.n_qubits >= 20: # L=8
            p_auto, n_restarts, opt_shots, tol = 2, 5, 1000, 1e-9
            use_cvar = False
        else: 
            # L=2: MODO SNIPER + CVaR
            p_auto, n_restarts, opt_shots, tol = 5, 20, 5000, 1e-9
            max_iter = 5000 
            use_cvar = True # <--- ACTIVAMOS CVaR
        
        p_final = p_layers if p_layers > 1 else p_auto

        print(f"\n🏋️  QAOA LOCAL (CPU) | p={p_final} | Max Iters={max_iter} | {n_restarts} Reinicios")
        if use_cvar: print("   ☢️  MODO CVaR ACTIVADO (Optimizando el Top 10% de disparos)")
        
        ansatz = QAOAAnsatz(self.qiskit_hamiltonian, reps=p_final)
        ansatz.measure_all()
        transpiled_qc = transpile(ansatz, backend_local)
        
        best_global_energy = float('inf')
        best_global_params = None
        best_global_history = []

        for i in range(n_restarts):
            current_history = []
            
            def objective_function(params):
                try: bound_qc = transpiled_qc.assign_parameters(params)
                except: bound_qc = transpiled_qc.bind_parameters(params)
                
                try:
                    job = backend_local.run(bound_qc, shots=opt_shots)
                    counts = job.result().get_counts()
                except Exception:
                    # Penalizar fuertemente fallos de evaluación para evitar falsos mínimos.
                    penalty = 1e12
                    current_history.append(penalty)
                    return penalty
                
                # --- LÓGICA CVaR (Conditional Value at Risk) ---
                if use_cvar:
                    all_energies = []
                    for b, count in counts.items():
                        bs = b.replace(" ", "")[-self.n_qubits:]
                        en = self.compute_energy_from_bitstring(bs)
                        # Expandimos según el número de cuentas
                        all_energies.extend([en] * count)
                    
                    # Ordenamos de menor a mayor
                    all_energies.sort()
                    # Nos quedamos con el mejor 10% (alpha = 0.1)
                    alpha = 0.1
                    num_keep = max(1, int(len(all_energies) * alpha))
                    cvar_value = np.mean(all_energies[:num_keep])
                    
                    current_history.append(cvar_value)
                    return cvar_value
                
                # --- LÓGICA ESTÁNDAR (Valor Esperado) ---
                else:
                    total_en = 0; total_cts = 0
                    for b, c in counts.items():
                        bs = b.replace(" ", "")[-self.n_qubits:]
                        total_en += self.compute_energy_from_bitstring(bs) * c
                        total_cts += c
                    avg = total_en / total_cts if total_cts > 0 else 0
                    current_history.append(avg)
                    return avg

            # INICIALIZACIÓN RAMPA LINEAL
            if i == 0 and self.n_qubits < 20:
                print("   ⚡ Usando Inicialización Adiabática (Linear Ramp)...")
                betas = np.linspace(1.0, 0.0, p_final) * np.pi
                gammas = np.linspace(0.0, 1.0, p_final) * np.pi
                initial_point = []
                for k in range(p_final):
                    initial_point.append(betas[k])  
                    initial_point.append(gammas[k]) 
                initial_point = np.array(initial_point)
            else:
                initial_point = np.random.uniform(0, 2*np.pi, 2 * p_final)

            optimizer = COBYLA(maxiter=max_iter, tol=tol)
            res = optimizer.minimize(objective_function, initial_point)
            
            # Nota: Con CVaR, res.fun es la media del top 10%, no el promedio total.
            # Pero sigue sirviendo para elegir el mejor set de parámetros.
            if res.fun < best_global_energy:
                best_global_energy = res.fun
                best_global_params = res.x
                best_global_history = current_history
                if n_restarts > 1: 
                    print(f"   ✅ Restart {i+1}: Récord -> {best_global_energy:.6f}")

        if best_global_params is None:
            raise RuntimeError("QAOA no produjo parámetros válidos durante la optimización.")

        # --- FASE FINAL ---
        probs_dict = {}
        
        if ibm_token and IBM_AVAILABLE:
            print("\n☁️  CONECTANDO A IBM QUANTUM...")
            try:
                service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
                backend_ibm = service.least_busy(operational=True, simulator=False, min_num_qubits=self.n_qubits)
                print(f"   🤖 Backend: {backend_ibm.name}")
                
                ibm_qc = transpile(ansatz, backend_ibm, optimization_level=3)
                try: final_qc = ibm_qc.assign_parameters(best_global_params)
                except: final_qc = ibm_qc.bind_parameters(best_global_params)

                print("   🚀 Enviando Job...")
                job = backend_ibm.run(final_qc, shots=4000)
                print(f"   🆔 Job ID: {job.job_id()}")
                
                result = job.result()
                counts = result.get_counts()
                for b, c in counts.items():
                    bs = b.replace(" ", "")[-self.n_qubits:]
                    probs_dict[bs] = c / 4000
                    
            except Exception as e:
                print(f"❌ Error IBM: {e}. Usando local.")
                probs_dict = self._run_local_final(transpiled_qc, best_global_params, backend_local)
        else:
            print("   🏠 Ejecutando tirada final Local...")
            probs_dict = self._run_local_final(transpiled_qc, best_global_params, backend_local)

        best_bs, min_en = self._smart_select(probs_dict)
        try: self.plotter.plot_optimization(best_global_history, solver_name="QAOA")
        except: pass

        return {
            'bitstring': best_bs, 'energy': min_en, 'costs': best_global_history, 
            'repaired_sequence': self.decode_solution(best_bs), 
            'repaired_cost': min_en, 'probs': probs_dict
        }

    def _run_local_final(self, qc, params, backend):
        self._ensure_qiskit("La ejecución final QAOA")
        try: final_qc = qc.assign_parameters(params)
        except: final_qc = qc.bind_parameters(params)
        # SUPER SHOTS PARA L=2 PARA REDUCIR RUIDO ESTADÍSTICO
        shots = 200000 if self.n_qubits < 20 else 5000
        job = backend.run(final_qc, shots=shots)
        counts = job.result().get_counts()
        probs = {}
        for b, c in counts.items():
            bs = b.replace(" ", "")[-self.n_qubits:]
            probs[bs] = c / shots
        return probs
    def solve_vqe_qiskit(self, layers=1, max_iter=100):
        """
        Resuelve usando VQE con Evolución Diferencial (Algoritmo Genético).
        No requiere dependencias externas de Qiskit Algorithms.
        """
        self._ensure_qiskit("El solver VQE")
        print(f"\n🧬 INICIANDO VQE (Differential Evolution) | Layers={layers}")
        
        # 1. Imports locales para no romper nada arriba
        from qiskit.circuit.library import TwoLocal
        from qiskit import transpile
        try:
            from scipy.optimize import differential_evolution
        except ImportError as e:
            raise ImportError(
                "VQE requiere scipy (scipy.optimize.differential_evolution)."
            ) from e
        import sys

        # 2. Configuración del Backend (Simulador Local)
        # Usamos el mismo simulador que ya tienes configurado en la clase
        backend = AerSimulator(method='statevector')
        
        # 3. Crear el Ansatz (Circuito Variacional)
        # TwoLocal es el estándar: Ry para rotar, CZ para entrelazar
        ansatz = TwoLocal(self.n_qubits, ['ry'], 'cz', reps=layers, entanglement='full')
        ansatz.measure_all()
        
        # 4. Transpilar una sola vez para velocidad
        print(f"   ⚙️  Transpilando circuito ({ansatz.num_parameters} parámetros)...")
        transpiled_qc = transpile(ansatz, backend)
        
        # Historial para la gráfica
        history = []

        # 5. Definir la Función de Coste (Objective Function)
        # Esta función es la que el algoritmo genético va a llamar miles de veces
        def objective_function(params):
            # Asignar los parámetros genéticos al circuito
            try:
                bound_qc = transpiled_qc.assign_parameters(params)
            except:
                bound_qc = transpiled_qc.bind_parameters(params)
            
            # Ejecutar simulación
            # Usamos menos shots durante la optimización para ir rápido
            job = backend.run(bound_qc, shots=1000) 
            counts = job.result().get_counts()
            
            # Calcular energía promedio (Expectation Value)
            total_energy = 0
            total_counts = 0
            
            for bitstring, count in counts.items():
                # Asegurar formato correcto del bitstring
                bs = bitstring.replace(" ", "")[-self.n_qubits:]
                energy = self.compute_energy_from_bitstring(bs)
                total_energy += energy * count
                total_counts += count
                
            avg_energy = total_energy / total_counts
            
            # Guardar para historial (imprimir cada 500 evaluaciones para no saturar)
            history.append(avg_energy)
            if len(history) % 500 == 0:
                print(f"      Genética en progreso... Eval #{len(history)}: E = {avg_energy:.4f}")
                
            return avg_energy

        # 6. Configurar y Ejecutar Evolución Diferencial
        # Bounds: Cada ángulo puede ir de 0 a 2pi
        bounds = [(0, 2*np.pi) for _ in range(ansatz.num_parameters)]
        
        print(f"   🚀 Lanzando población de mutantes (paciencia, esto tarda)...")
        # workers=-1 usa todos los núcleos de tu CPU
        result = differential_evolution(
            objective_function, 
            bounds, 
            maxiter=max_iter, 
            popsize=10, # Tamaño de población (bájalo si va muy lento)
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=True # Muestra progreso en consola
        )

        print(f"   🏆 VQE Terminado. Mejor Energía: {result.fun:.6f}")

        # 7. Recuperar la mejor solución encontrada
        best_params = result.x
        
        # Tirada final con muchos shots para tener una buena distribución de probabilidad
        try:
            final_qc = transpiled_qc.assign_parameters(best_params)
        except:
            final_qc = transpiled_qc.bind_parameters(best_params)
            
        final_job = backend.run(final_qc, shots=10000)
        final_counts = final_job.result().get_counts()
        
        probs_dict = {}
        for b, c in final_counts.items():
            bs = b.replace(" ", "")[-self.n_qubits:]
            probs_dict[bs] = c / 10000

        # Seleccionar el ganador
        best_bs, min_en = self._smart_select(probs_dict)

        # Intentar graficar convergencia
        try:
            self.plotter.plot_optimization(history, solver_name="VQE_DiffEvol")
        except: pass

        return {
            'bitstring': best_bs, 
            'energy': min_en, 
            'costs': history, 
            'repaired_sequence': self.decode_solution(best_bs), 
            'repaired_cost': min_en, 
            'probs': probs_dict
        }

    def _smart_select(self, probs_dict):
        best_bs, min_en = None, float('inf')
        for bs in probs_dict.keys():
            en = self.compute_energy_from_bitstring(bs)
            if en < min_en: min_en, best_bs = en, bs
        if best_bs is None: return '0'*self.n_qubits, 0.0
        return best_bs, min_en


    def solve_quantum_annealing(self, num_reads=1000):
        """
        Resuelve el problema usando Simulated Annealing (dwave-neal).
        Simula el comportamiento de un Quantum Annealer (D-Wave) en tu CPU.
        Excelente para problemas grandes donde QAOA/VQE son lentos.
        """
        print(f"\n🏔️  SOLVING WITH SIMULATED ANNEALING (NEAL)...")
        
        try:
            import dimod
            from neal import SimulatedAnnealingSampler
        except ImportError:
            print("❌ Error: Necesitas instalar 'dwave-ocean-sdk' o 'dwave-neal'.")
            print("   Ejecuta: pip install dwave-neal dimod")
            return {}

        # 1. Convertir tu Hamiltoniano (Pauli Z) a un Modelo Ising (h, J)
        h = {} # Términos lineales (Z en un solo qubit)
        J = {} # Términos cuadráticos (Z en dos qubits)
        offset = 0.0

        print("   ⚙️  Convirtiendo Hamiltoniano a formato Ising/QUBO...")
        
        # Recorremos tus términos de Pauli ya construidos
        for coeff, pauli_str in self.pauli_terms:
            # Contar cuántas Z hay en la cadena y dónde están
            z_indices = [i for i, char in enumerate(pauli_str) if char == 'Z']
            
            if len(z_indices) == 0:
                # Es el término identidad (Offset global)
                offset += coeff
            elif len(z_indices) == 1:
                # Término lineal (h) -> Bias local
                i = z_indices[0]
                h[i] = h.get(i, 0.0) + coeff
            elif len(z_indices) == 2:
                # Término cuadrático (J) -> Acoplamiento
                i, j = z_indices[0], z_indices[1]
                if i > j: i, j = j, i # Ordenar para consistencia
                J[(i, j)] = J.get((i, j), 0.0) + coeff
            else:
                pass # Términos de orden superior (>2-local) no soportados nativamente por Ising estándar

        # Crear el Binary Quadratic Model (BQM)
        bqm = dimod.BinaryQuadraticModel(h, J, offset, dimod.SPIN) # SPIN = Ising (-1, +1)
        
        # 2. Configurar el Sampler (Simulador Local)
        print("   💻 Iniciando Neal SimulatedAnnealingSampler...")
        sampler = SimulatedAnnealingSampler()

        # 3. Ejecutar el Annealing
        print(f"   🚀 Ejecutando {num_reads} lecturas (reads)...")
        sampleset = sampler.sample(bqm, num_reads=num_reads)

        # 4. Procesar Resultados
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy
        
        # Convertir diccionario de espines Ising (-1, +1) a bitstring (1, 0)
        # Ising +1 (Up)   -> Bit 0
        # Ising -1 (Down) -> Bit 1
        bitstring_list = ['0'] * self.n_qubits
        for qubit_idx, spin_val in best_sample.items():
            if qubit_idx < self.n_qubits:
                bitstring_list[qubit_idx] = '0' if spin_val == 1 else '1'
        
        best_bs = "".join(bitstring_list)
        
        # Construir distribución de probabilidades para el reporte
        probs_dict = {}
        total_counts = sum(sampleset.record.num_occurrences)
        for data in sampleset.data(['sample', 'num_occurrences']):
            bs_list = ['0'] * self.n_qubits
            for q_idx, s_val in data.sample.items():
                 if q_idx < self.n_qubits:
                    bs_list[q_idx] = '0' if s_val == 1 else '1'
            bs_str = "".join(bs_list)
            probs_dict[bs_str] = data.num_occurrences / total_counts

        decoded_seq = self.decode_solution(best_bs)
        
        print(f"   🏆 Mejor solución encontrada: {decoded_seq} (E = {best_energy:.6f})")

        return {
            'bitstring': best_bs, 
            'energy': best_energy, 
            'costs': [], # Simulated annealing no devuelve historial de costes paso a paso fácil
            'repaired_sequence': decoded_seq, 
            'repaired_cost': best_energy, 
            'probs': probs_dict
        }

