import os
import matplotlib
# CRÍTICO: 'Agg' permite guardar gráficas sin pantalla (evita errores en clúster)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

class ProteinPlotter:
    """Clase para manejar todas las visualizaciones del diseño de proteínas"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # --- FUNCIÓN DE OPTIMIZACIÓN (Tu versión favorita con el punto rojo en el mínimo) ---
    def plot_optimization(self, costs: list, solver_name: str, tick_size: int = 14):
        """
        Plots the optimization convergence (costs vs iterations).
        Highlights the MINIMUM cost found.
        """
        if not costs:
            print(f"⚠️ Not enough data to plot {solver_name} (len=0)")
            return
        
        iterations = np.arange(len(costs))
        plt.figure(figsize=(10, 6))
        
        # Main line
        plt.plot(iterations, costs, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.8, label='Cost')
        
        # Highlight the BEST (minimum) cost
        best_idx = np.argmin(costs)
        best_cost = costs[best_idx]
        plt.scatter(best_idx, best_cost, color='red', s=250, zorder=5,
                    edgecolor='darkred', linewidth=3, label=f'Minimum energy')
        
        plt.tick_params(axis='both', which='major', labelsize=tick_size)
        plt.tick_params(axis='both', which='minor', labelsize=tick_size-2)
        
        # Styling
        plt.xlabel('Iterations', fontsize=15, fontweight='bold')
        plt.ylabel('Energy / Cost', fontsize=15, fontweight='bold')
        
        title = f'{solver_name} Convergence'
        if solver_name.upper() == "QAOA":
            title = "QAOA Energy Convergence"
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save
        filename = f'{solver_name.lower()}_optimization_convergence.png'
        output_path = os.path.join(self.output_dir, filename)
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ {solver_name} convergence plot saved: {output_path}")
        except Exception as e:
            print(f"⚠️ Error saving optimization plot: {e}")
        finally:
            plt.close()

    # Alias por si acaso quantum_engine llama a plot_convergence
    def plot_convergence(self, costs, title="QAOA"):
        self.plot_optimization(costs, solver_name=title)

    # --- FUNCIÓN DE HELICAL WHEEL (Restaurada con tus parámetros exactos) ---
    def plot_alpha_helix_wheel(self, sequence: str, membrane_mode: str = 'wheel', 
                                    wheel_phase_deg: float = 0.0, 
                                    wheel_halfwidth_deg: float = 90.0, 
                                    auto_align: bool = False):
                """
                Plot alpha helix wheel visualization.
                - FIXED LOGIC: Background split is purely geometric based on wheel_halfwidth_deg.
                - Lipids (Brown) are centered at 0 degrees (Right side).
                - Water (Blue) is the rest.
                """
                import matplotlib.patches as mpatches
                import matplotlib.pyplot as plt
                import numpy as np
                import os
                import math

                if not sequence or sequence.count('X') == len(sequence):
                    print(f"Warning: Invalid sequence for helix wheel: {sequence}")
                    return
                
                print(f"Plotting alpha helix wheel for sequence: {sequence}")
                
                # --- 1. CLASIFICACIÓN Y COLORES (ESTILO ORIGINAL) ---
                nonpolar = set(['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P', 'C'])
                polar = set(['S', 'T', 'N', 'Q', 'Y', 'G'])
                negative = set(['D', 'E'])
                positive = set(['K', 'R', 'H'])
                
                color_map = {}
                for aa in sequence:
                    if aa in negative: color_map[aa] = 'red'
                    elif aa in positive: color_map[aa] = 'blue'
                    elif aa in nonpolar: color_map[aa] = '#8B4513' # Marrón original
                    elif aa in polar: color_map[aa] = 'green'
                    else: color_map[aa] = 'gray'

                # --- 2. CÁLCULO DE ROTACIÓN ---
                angle_increment = np.deg2rad(100.0)
                # Fase 0 significa empezar a la derecha. Ajustamos según wheel_phase_deg.
                rotation_offset = np.deg2rad(wheel_phase_deg)

                # --- 3. GENERAR COORDENADAS ---
                base_radius = 1.0
                layer_offset = 0.35
                overlap_threshold = 18
                
                xs = []
                ys = []
                radii = [] 
                
                for i in range(len(sequence)):
                    theta = i * angle_increment + rotation_offset
                    layer = i // overlap_threshold
                    r = base_radius + (layer * layer_offset)
                    radii.append(r)
                    xs.append(r * np.cos(theta))
                    ys.append(r * np.sin(theta))
                
                max_r = max(radii) if radii else base_radius
                plot_limit = max_r + 1.5

                # --- 4. CÁLCULO DE LA LÍNEA VERTICAL (GEOMETRÍA FIJA) ---
                # Si halfwidth es 90, la membrana va de -90 a +90 (todo el lado derecho).
                # La línea divisoria es x = 0.
                # Si halfwidth es 60, la membrana es más estrecha.
                # La línea es x = cos(90 - (90-60)) ... simplificado:
                # Para halfwidth < 90, la línea está en X positivo.
                # Para halfwidth > 90, la línea está en X negativo.
                
                # Calculamos la coordenada X de la frontera
                # Asumimos simetría vertical. La frontera es una línea vertical aproximada
                # para mantener el estilo rectangular que te gusta.
                
                if wheel_halfwidth_deg >= 180:
                    interface_x = -plot_limit - 1.0 # Todo lípidos
                elif wheel_halfwidth_deg <= 0:
                    interface_x = plot_limit + 1.0  # Todo agua
                else:
                    # Proyección en el eje X del ángulo de corte
                    # Si ancho=90, ángulo corte=90, cos(90)=0 -> x=0
                    # Si ancho=60, ángulo corte=60, cos(60)=0.5 -> x>0
                    # Pero queremos el borde izquierdo de la zona marrón.
                    # La zona marrón está centrada en 0 grados.
                    # Su borde está en +halfwidth y -halfwidth.
                    # Si aproximamos con una línea recta vertical (como tu dibujo original):
                    # La línea debe estar en x = cos(180 - halfwidth)? No.
                    # Simplemente: X = 0 es mitad y mitad.
                    # X positivo = menos lípidos. X negativo = más lípidos.
                    # Fórmula aproximada para visualización rectangular:
                    # Si halfwidth=90 -> x=0. Si halfwidth=180 -> x=-limit. Si halfwidth=0 -> x=limit.
                    
                    # Interpolación lineal simple para visualización
                    ratio = (90.0 - wheel_halfwidth_deg) / 90.0 
                    interface_x = ratio * plot_limit

                plt.figure(figsize=(9, 9))
                ax = plt.gca()
                
                # --- 5. DIBUJAR FONDO (ESTILO ORIGINAL) ---
                
                # A. AGUA (Izquierda / Fondo Total)
                rect_water = mpatches.Rectangle(
                    xy=(-plot_limit, -plot_limit),
                    width=2 * plot_limit, 
                    height=2 * plot_limit,
                    facecolor='#CCEEFF', # Azul clarito original
                    alpha=0.6, zorder=0
                )
                ax.add_patch(rect_water)

                # B. LÍPIDOS (Derecha, recortado por interface_x)
                if interface_x < plot_limit:
                    width_lipid = plot_limit - interface_x
                    rect_lipids = mpatches.Rectangle(
                        xy=(interface_x, -plot_limit),
                        width=width_lipid, 
                        height=2 * plot_limit,
                        facecolor='#FFE4C4', # 'Bisque' (Marrón muy claro original)
                        alpha=0.6, zorder=0
                    )
                    ax.add_patch(rect_lipids)
                    
                    # Línea divisoria
                    ax.axvline(x=interface_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

                # Etiquetas (Centradas en sus zonas aproximadas)
                center_water = (-plot_limit + interface_x) / 2
                center_lipid = (interface_x + plot_limit) / 2
                
                if center_water > -plot_limit:
                    ax.text(center_water, -plot_limit * 0.9, 'Water Phase', 
                            ha='center', va='bottom', fontsize=16, color='#00008B', weight='bold')
                
                if center_lipid < plot_limit:
                    ax.text(center_lipid, -plot_limit * 0.9, 'Lipid Phase', 
                            ha='center', va='bottom', fontsize=16, color='#8B4513', weight='bold')

                # --- 6. DIBUJAR ESTRUCTURA (ESTILO ORIGINAL) ---
                for i in range(len(sequence) - 1):
                    plt.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], 
                            color='gray', alpha=0.5, linewidth=1.5, zorder=2) # Gris original

                for i, aa in enumerate(sequence):
                    # Círculos grandes y bordes negros
                    plt.scatter(xs[i], ys[i], s=800, color=color_map[aa], 
                            edgecolors='black', linewidth=1.5, zorder=3)
                    # Letra AA
                    plt.text(xs[i], ys[i], aa, ha='center', va='center', 
                            fontsize=14, weight='bold', color='white', zorder=4)
                    
                    # Número de residuo fuera
                    label_pos_r = radii[i] + 0.3
                    ang_i = i * angle_increment + rotation_offset
                    xi = label_pos_r * np.cos(ang_i)
                    yi = label_pos_r * np.sin(ang_i)
                    plt.text(xi, yi, f"{i+1}", ha='center', va='center', 
                            fontsize=11, color='black', weight='bold', zorder=5)

                # Círculo guía punteado
                circle = plt.Circle((0, 0), base_radius, color='gray', fill=False, 
                                    linestyle='--', alpha=0.3, zorder=1)
                ax.add_artist(circle)
                
                ax.set_aspect('equal')
                ax.set_xlim(-plot_limit, plot_limit)
                ax.set_ylim(-plot_limit, plot_limit)
                ax.axis('off')
                plt.title(f'Alpha-Helix Wheel (Length: {len(sequence)})', pad=20, fontsize=16)
                
                output_path = os.path.join(self.output_dir, 'alpha_helix_wheel.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Alpha helix wheel plot saved as {output_path}")
    # --- OTRAS FUNCIONES (Probabilidades, Circuitos) ---
    def plot_prob_with_sequences(self, probs: np.ndarray, decoder_fn, n_qubits: int, 
                                solver_name: str = "QAOA", top_k: int = 20):
        
        # Adaptador para cuando probs es un diccionario (nuestro caso MPS)
        if isinstance(probs, dict):
            # Convertir dict a formato compatible
            sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
            sequences = [decoder_fn(bs) for bs, _ in sorted_items]
            sorted_probs = [p for _, p in sorted_items]
        else:
            # Lógica original para arrays
            if len(probs) == 0 or np.all(probs == 0): return
            top_k = min(top_k, len(probs))
            sorted_indices = np.argsort(-probs)[:top_k]
            sorted_probs = probs[sorted_indices]
            sequences = [decoder_fn(format(idx, f'0{n_qubits}b')) for idx in sorted_indices]

        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(sequences)), sorted_probs, color='steelblue', alpha=0.8, edgecolor='navy')
        if len(bars) > 0:
            bars[0].set_color('gold')
            bars[0].set_edgecolor('darkorange')
        
        plt.xlabel(f'Amino Acid Sequences (Top {len(sequences)})', fontsize=16, fontweight='bold')
        plt.ylabel('Probability', fontsize=16, fontweight='bold')
        plt.xticks(range(len(sequences)), sequences, rotation=90, fontsize=14)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{solver_name.lower()}_probability_plot.png')
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Probability plot saved as {output_path}")
        except: pass
        finally: plt.close()

    def save_qiskit_circuit(self, circuit, filename: str):
        """Save Qiskit circuit to a PNG/TXT file."""
        output_path = os.path.join(self.output_dir, filename)
        try:
            from qiskit.visualization import circuit_drawer
            try:
                circuit_drawer(circuit, output='mpl', filename=output_path)
            except:
                # Fallback a texto si falla mpl
                txt_path = output_path.replace('.png', '.txt')
                with open(txt_path, 'w') as f:
                    f.write(str(circuit_drawer(circuit, output='text')))
        except: pass

