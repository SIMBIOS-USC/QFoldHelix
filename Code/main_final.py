import os
import time
import argparse
import traceback
import matplotlib
matplotlib.use('Agg')  # Forzar backend sin GUI

def main():
    parser = argparse.ArgumentParser(description='Quantum protein sequence design.')
    parser.add_argument('-L', '--length', type=int, default=4, help='Sequence length.')
    parser.add_argument('-R', '--residues', type=str, default="V,Q,L,R", help='Amino acids to use, comma-separated.')
    parser.add_argument('-b', '--backend', type=str, default='qiskit', choices=['pennylane', 'qiskit'], help='Quantum backend to use.')
    
    # Solver options
    parser.add_argument('--solver', type=str, default='qaoa', choices=['qaoa', 'vqe', 'classical', 'annealing'], help='Solver to use.')
    
    parser.add_argument('--shots', type=int, default=10000000, help='Number of shots for quantum simulation.')
    parser.add_argument('--membrane', type=str, help='Membrane span (e.g., 1:4)')
    parser.add_argument('--membrane_positions', type=str, help='Membrane positions (e.g., 0,2,5)')
    parser.add_argument('--membrane_mode', type=str, default='wheel', choices=['span', 'set', 'wheel'], help='Mode for defining membrane positions.')
    parser.add_argument('--wheel_phase_deg', type=float, default=-50.0, help='Phase angle for helical wheel in degrees.')
    parser.add_argument('--wheel_halfwidth_deg', type=float, default=120.0, help='Half-width of the membrane sector in degrees for helical wheel.')
    
    # --- Argumentos de pesos (Lambdas) ---
    parser.add_argument('--lambda_env', type=float, default=-10, help='Weight of the environment preference term.')# poner 0.1 para el polar
    parser.add_argument('--lambda_charge', type=float, default=3, help='Weight of the membrane charge term.')
    parser.add_argument('--lambda_mu', type=float, default=0.2, help='Weight of the hydrophobic moment term.')
    parser.add_argument('--lambda_local', type=float, default=1, help='Weight of the local preference terms.')
    parser.add_argument('--lambda_pairwise', type=float, default=0.1, help='Weight of the pairwise interaction term.')
    parser.add_argument('--lambda_helix_pairs', type=float, default=1, help='Weight of the helix pair propensity term.') # Bajado de 50 a 5.0 (valor razonable)
    parser.add_argument('--lambda_electrostatic', type=float, default=0.5, help='Weight of the general electrostatics term.') # Bajado de 50 a 5.0
    parser.add_argument('--max_interaction_dist', type=int, default=1, help='Maximum sequence distance for pairwise interactions.') # Cambiado default a 4 para captar interacciones de hélice
    parser.add_argument('--membrane_charge', type=str, default='pos', choices=['neu', 'neg', 'pos'], help='Charge of the membrane.')
    
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files.')
    parser.add_argument('--use_statevector', action='store_true', default=False, help='Use statevector instead of shots.')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Procesar aminoácidos
    aa_list = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W","Y", "V"]
    if args.residues:
        s = args.residues.upper().strip()
        if len(s) > 1 and ',' in s:
            aa_list = [t.strip() for t in s.split(",") if t.strip()]
        else:
            aa_list = [c for c in s if c.strip()]

    # Procesar membrana
    mem_span = None
    if args.membrane:
        try:
            a, b = args.membrane.split(":")
            mem_span = (int(a), int(b))
        except Exception:
            print("Invalid --membrane format. Use start:end, e.g. 1:4")
            return

    mem_positions = None
    if args.membrane_positions:
        try:
            mem_positions = [int(t) for t in args.membrane_positions.split(',') if t.strip()]
        except Exception:
            print("Invalid --membrane_positions. Use comma-separated indices, e.g. 0,2,5")
            return

    result = {}

    print(f"🚀 Iniciando diseño (L={args.length}, Solver={args.solver})...")
    start_time = time.time()
    
    try:
        # Importaciones diferidas para permitir `--help` sin dependencias pesadas.
        from src.quantum_engine import QuantumProteinDesign
        from src.reports import save_energy_results, print_top_sequences_table

        if args.backend == 'pennylane' and args.solver in {'qaoa', 'vqe'}:
            print("❌ El backend 'pennylane' no está implementado para --solver qaoa/vqe en este entrypoint.")
            print("   Usa --backend qiskit o cambia a --solver classical/annealing.")
            return

        # 1. Crear el objeto Designer
        designer = QuantumProteinDesign(
            sequence_length=args.length,
            amino_acids=aa_list,
            quantum_backend=args.backend,
            shots=args.shots,
            membrane_span=mem_span,
            membrane_charge=args.membrane_charge,
            lambda_charge=args.lambda_charge,
            lambda_env=args.lambda_env,
            lambda_mu=args.lambda_mu,
            lambda_local=args.lambda_local,
            lambda_pairwise=args.lambda_pairwise,
            lambda_helix_pairs=args.lambda_helix_pairs,
            lambda_electrostatic=args.lambda_electrostatic,
            
            # --- PASAMOS EL NUEVO ARGUMENTO ---
            # ----------------------------------
            
            max_interaction_dist=args.max_interaction_dist,
            membrane_positions=mem_positions,
            membrane_mode=args.membrane_mode,
            wheel_phase_deg=args.wheel_phase_deg,
            wheel_halfwidth_deg=args.wheel_halfwidth_deg,
            solver=args.solver,
            output_dir=args.output_dir,
            use_statevector=args.use_statevector
        )
        
        # 2. EJECUTAR EL SOLVER
        if args.solver == 'classical':
            result = designer.solve_classical_brute_force()
        elif args.solver == 'vqe':
            result = designer.solve_vqe_qiskit()
        elif args.solver == 'annealing':
            result = designer.solve_quantum_annealing(num_reads=2000)
        else: # qaoa
            result = designer.solve_qaoa_qiskit()
        result.setdefault('solver', args.solver)

        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n✅ Optimization complete!")
        
        # 3. Mostrar Resultados
        if args.solver == 'classical':
            print("\n🏆 Solución Clásica:")
            print(f"Secuencia: {result.get('repaired_sequence', result.get('sequence', 'N/A'))}")
            print(f"Energía: {result.get('energy', float('inf')):.6f}")
        else:
            print(f"\n⚛️ Solución Cuántica ({args.solver.upper()}):")
            print(f"Secuencia Reparada: {result.get('repaired_sequence', 'N/A')}")
            print(f"Energía Final: {result.get('repaired_cost', float('inf')):.6f}")

        # 4. Guardar Logs
        seq_log = result.get('repaired_sequence', result.get('sequence', 'N/A'))
        log_entry = f"Solver: {args.solver} | Time: {execution_time:.4f}s | Sequence: {seq_log}\n"
        
        with open(os.path.join(args.output_dir, "execution_log.txt"), "a") as log_file:
            log_file.write(log_entry)
        
        # 5. Generar Reportes Detallados y Gráficos
        save_energy_results(designer, result, args.solver, args.output_dir)
        print_top_sequences_table(designer, result)
        
        if result.get('costs'):
            designer.plotter.plot_optimization(result['costs'], solver_name=args.solver.upper())

        if seq_log != 'N/A' and 'X' not in seq_log:
             designer.plotter.plot_alpha_helix_wheel(
                sequence=seq_log,
                wheel_phase_deg=args.wheel_phase_deg,
                wheel_halfwidth_deg=args.wheel_halfwidth_deg,
                membrane_mode=args.membrane_mode
            )

    except Exception:
        print("❌ Error durante la ejecución:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
