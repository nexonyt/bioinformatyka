import os
import time
import numpy as np
import matplotlib.pyplot as plt
from main import GeneticAlgorithmMSA 

def read_fasta(filepath):
    """Prosty parser plików FASTA."""
    sequences = []
    with open(filepath, 'r') as f:
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line
        if seq:
            sequences.append(seq)
    return sequences

def generate_protein_instances(folder, filename_prefix, num_seqs, base_len, mut_rate, gap_rate):
    """Pomocniczy generator ukryty w funkcji, by nie blokował głównego skryptu."""
    os.makedirs(folder, exist_ok=True)
    aminos = ['A', 'C', 'G', 'N', 'P', 'S', 'T']
    ancestor = [random.choice(aminos) for _ in range(base_len)]
    
    sequences = []
    for _ in range(num_seqs):
        seq = []
        for char in ancestor:
            if random.random() < mut_rate:
                seq.append(random.choice([a for a in aminos if a != char]))
            elif random.random() < gap_rate:
                continue
            else:
                seq.append(char)
        sequences.append("".join(seq))
        
    filepath = os.path.join(folder, f"{filename_prefix}.fasta")
    with open(filepath, 'w') as f:
        for i, s in enumerate(sequences):
            f.write(f">seq_{i}\n{s}\n")
    print(f"Zweryfikowano/Wygenerowano instancję testową: {filepath}")

def run_parameter_test(dataset_folder, pop_sizes, num_runs=5, generations=50):
    """Testuje wpływ rozmiaru populacji na wyniki i rysuje wykres z tabelą."""
    print(f"\n[Rozpoczynam testowanie algorytmu genetycznego na sekwencjach białkowych]")
    print(f"Folder z danymi: {dataset_folder}")
    
    fasta_files = [f for f in os.listdir(dataset_folder) if f.endswith('.fasta')]
    if not fasta_files:
        print("Brak plików .fasta! Wygeneruj je najpierw.")
        return

    results_summary = {}

    for pop_size in pop_sizes:
        print(f"\n--- Badany parametr: ROZMIAR POPULACJI = {pop_size} ---")
        
        all_final_scores = []
        all_times = []
        convergence_history = np.zeros(generations) 

        for file in fasta_files:
            filepath = os.path.join(dataset_folder, file)
            sequences = read_fasta(filepath)
            
            for run in range(num_runs):
                start_time = time.time()
                
                # Uruchomienie zaawansowanego GA (BLOSUM62)
                ga = GeneticAlgorithmMSA(sequences, pop_size=pop_size, generations=generations, mutation_rate=0.2)
                best_alignment, history = ga.run()
                
                end_time = time.time()
                
                final_score = history[-1]
                all_final_scores.append(final_score)
                all_times.append(end_time - start_time)
                convergence_history += np.array(history)
                
        total_runs_per_param = len(fasta_files) * num_runs
        avg_convergence = convergence_history / total_runs_per_param
        
        results_summary[pop_size] = {
            'avg_score': np.mean(all_final_scores),
            'std_dev': np.std(all_final_scores),
            'max_score': np.max(all_final_scores),
            'avg_time': np.mean(all_times),
            'convergence': avg_convergence
        }

    # --- Generowanie Wykresu Zbieżności ---
    print("\n[Rysowanie wykresu zbieżności...]")
    plt.figure(figsize=(10, 6))
    for pop_size, data in results_summary.items():
        plt.plot(range(generations), data['convergence'], label=f'Populacja = {pop_size}', linewidth=2)
        
    plt.title('Wpływ rozmiaru populacji na SP-score (Białka, BLOSUM62)')
    plt.xlabel('Generacja')
    plt.ylabel('Średni SP-score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    chart_filename = 'wykres_zbieznosc_populacja.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    print(f"[!] Gotowy wykres zapisano do pliku: {chart_filename}")
    plt.show()

    # --- Drukowanie Tabeli Wyników ---
    print("\n" + "="*80)
    print("GOTOWA TABELA DO SPRAWOZDANIA (WARIANT ROZSZERZONY - BIAŁKA)")
    print("="*80)
    print(f"{'Populacja':<10} | {'Śr. Wynik (SP)':<15} | {'Najlepszy Wynik':<15} | {'Odchylenie Std':<15} | {'Śr. Czas [s]':<15}")
    print("-" * 80)
    for pop_size, data in results_summary.items():
        print(f"{pop_size:<10} | {data['avg_score']:<15.2f} | {data['max_score']:<15.2f} | {data['std_dev']:<15.2f} | {data['avg_time']:<15.2f}")
    print("="*80)

if __name__ == "__main__":
    import random # potrzebne do generatora
    target_folder = 'test_data'
    
    # 1. KROK: Upewniamy się, że pliki wejściowe istnieją
    print("[1/2] Przygotowanie instancji białkowych...")
    generate_protein_instances(target_folder, 'protein_easy', num_seqs=5, base_len=40, mut_rate=0.05, gap_rate=0.05)
    generate_protein_instances(target_folder, 'protein_medium', num_seqs=6, base_len=60, mut_rate=0.15, gap_rate=0.10)
    generate_protein_instances(target_folder, 'protein_hard', num_seqs=7, base_len=80, mut_rate=0.25, gap_rate=0.15)
    
    # 2. KROK: Uruchomienie właściwych testów
    print("\n[2/2] Odpalanie właściwego potoku testowego...")
    # num_runs=3, generations=40 dla szybkiego testu, zwiększ przed oddaniem
    run_parameter_test(target_folder, pop_sizes=[10, 30, 50], num_runs=3, generations=40)