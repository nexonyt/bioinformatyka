import random

class GeneticAlgorithmMSA:
    def __init__(self, sequences, pop_size=50, generations=100, mutation_rate=0.2, crossover_rate=0.7):
        self.sequences = sequences
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Kary za luki dla konwencji BLAST/NCBI (strona 6 instrukcji)
        self.gap_open = -12
        self.gap_ext = -1
        
        # Definicja macierzy podstawień BLOSUM62 (strona 4 instrukcji)
        self.amino_acids = ['A', 'C', 'G', 'N', 'P', 'S', 'T']
        self.blosum62 = {
            'A': {'A': 4,  'C': 0,  'G': 0,  'N': -2, 'P': -1, 'S': 1,  'T': 0},
            'C': {'A': 0,  'C': 9,  'G': -3, 'N': -3, 'P': -3, 'S': -1, 'T': -1},
            'G': {'A': 0,  'C': -3, 'G': 6,  'N': 0,  'P': -2, 'S': 0,  'T': -2},
            'N': {'A': -2, 'C': -3, 'G': 0,  'N': 6,  'P': -2, 'S': 1,  'T': 0},
            'P': {'A': -1, 'C': -3, 'G': -2, 'N': -2, 'P': 7,  'S': -1, 'T': -1},
            'S': {'A': 1,  'C': -1, 'G': 0,  'N': 1,  'P': -1, 'S': 4,  'T': 1},
            'T': {'A': 0,  'C': -1, 'G': -2, 'N': 0,  'P': -1, 'S': 1,  'T': 5}
        }

    def _get_substitution_score(self, a1, a2):
        """Zwraca wartość z macierzy BLOSUM62 dla pary aminokwasów."""
        """
        Funkcja pomocnicza sprawdzająca, ile punktów (dodatnich lub ujemnych)
        należy przyznać za zestawienie dwóch konkretnych aminokwasów.
        """
        # Jeśli któryś aminokwas wykracza poza uproszczoną macierz, domyślnie traktujemy jako mismatch
        if a1 in self.blosum62 and a2 in self.blosum62[a1]:
            return self.blosum62[a1][a2]
        return -4 if a1 != a2 else 4

    def calculate_sp_score(self, alignment):
        """
        Oblicza Sum-of-Pairs (SP-score) przy użyciu macierzy BLOSUM62
        oraz modelu kar afinicznych (Gap Open / Gap Extend).
        """
        num_seqs = len(alignment)
        length = len(alignment[0])
        total_score = 0
        
        # Porównanie każdej pary sekwencji (wiersz po wierszu)
        for i in range(num_seqs):
            for j in range(i + 1, num_seqs):
                seq1 = alignment[i]
                seq2 = alignment[j]
                
                in_gap1 = False
                in_gap2 = False
                
                for col in range(length):
                    c1 = seq1[col]
                    c2 = seq2[col]
                    
                    if c1 == '-' and c2 == '-':
                        # Dwie luki = 0 punktów (strona 2 instrukcji)
                        continue
                        
                    elif c1 == '-':
                        if not in_gap1:
                            total_score += self.gap_open
                            in_gap1 = True
                        else:
                            total_score += self.gap_ext
                        in_gap2 = False
                        
                    elif c2 == '-':
                        if not in_gap2:
                            total_score += self.gap_open
                            in_gap2 = True
                        else:
                            total_score += self.gap_ext
                        in_gap1 = False
                        
                    else:
                        # Pobranie punktacji z macierzy BLOSUM62 dla pary aminokwasów
                        total_score += self._get_substitution_score(c1, c2)
                        in_gap1 = False
                        in_gap2 = False
                        
        return total_score

    def initialize_population(self):
        """
        Tworzy punkt startowy (pokolenie zerowe) algorytmu.
        Wrzuca losowo luki ('-') do oryginalnych sekwencji, aż osiągną
        one wymaganą długość, po czym zwraca listę wylosowanych osobników.
        """
        population = []
        max_len = max(len(s) for s in self.sequences)
        target_len = int(max_len * 1.2) 
        
        for _ in range(self.pop_size):
            individual = []
            for seq in self.sequences:
                gaps_needed = target_len - len(seq)
                new_seq = list(seq)
                for _ in range(gaps_needed):
                    insert_idx = random.randint(0, len(new_seq))
                    new_seq.insert(insert_idx, '-')
                individual.append("".join(new_seq))
            population.append(individual)
            
        return population

    def remove_empty_columns(self, alignment):
        """
        Funkcja sprzątająca (optymalizacyjna). Usuwa kolumny, w których dla
        wszystkich sekwencji znajduje się wyłącznie znak luki, redukując długość.
        """
        num_seqs = len(alignment)
        length = len(alignment[0])
        cols_to_keep = []
        
        for col in range(length):
            if any(alignment[row][col] != '-' for row in range(num_seqs)):
                cols_to_keep.append(col)
                
        new_alignment = []
        for row in range(num_seqs):
            new_row = "".join(alignment[row][c] for c in cols_to_keep)
            new_alignment.append(new_row)
            
        return new_alignment

    def mutate(self, alignment):
        """
        Operator mutacji zapobiegający stagnacji algorytmu.
        Z określonym prawdopodobieństwem wybiera losową sekwencję, wycina w niej
        cały blok luk i przenosi go w inne, losowe miejsce.
        """
        if random.random() > self.mutation_rate:
            return alignment

        new_align = [list(seq) for seq in alignment]
        row = random.randint(0, len(new_align) - 1)
        
        gap_indices = [i for i, char in enumerate(new_align[row]) if char == '-']
        if not gap_indices:
            return alignment
            
        start_idx = random.choice(gap_indices)
        
        left = start_idx
        while left > 0 and new_align[row][left - 1] == '-':
            left -= 1
            
        right = start_idx
        while right < len(new_align[row]) - 1 and new_align[row][right + 1] == '-':
            right += 1
            
        block_length = right - left + 1
        
        del new_align[row][left:right + 1]
        
        insert_idx = random.randint(0, len(new_align[row]))
        for _ in range(block_length):
            new_align[row].insert(insert_idx, '-')
            
        return ["".join(seq) for seq in new_align]

    def crossover(self, parent1, parent2):
        """
        Operator krzyżowania łączący materiał genetyczny dwóch rozwiązań.
        Znajduje bezpieczny punkt odcięcia (gdzie zużyto tę samą liczbę aminokwasów)
        i skleja lewą połowę rodzica 1 z prawą połową rodzica 2.
        """
        num_seqs = len(parent1)
        
        def get_states(parent):
            states = {}
            current_counts = [0] * num_seqs
            for col in range(len(parent[0])):
                for row in range(num_seqs):
                    if parent[row][col] != '-':
                        current_counts[row] += 1
                state = tuple(current_counts)
                if state not in states:
                    states[state] = []
                states[state].append(col)
            return states

        states_p1 = get_states(parent1)
        states_p2 = get_states(parent2)
        
        common_states = set(states_p1.keys()).intersection(set(states_p2.keys()))
        valid_cuts = [s for s in common_states if sum(s) > 0 and sum(s) < sum(len(seq.replace("-", "")) for seq in parent1)]
        
        if not valid_cuts:
            return list(parent1)

        cut_state = random.choice(valid_cuts)
        cut_col_p1 = random.choice(states_p1[cut_state])
        cut_col_p2 = random.choice(states_p2[cut_state])
        
        child = []
        for i in range(num_seqs):
            left_part = parent1[i][:cut_col_p1 + 1]
            right_part = parent2[i][cut_col_p2 + 1:]
            child.append(left_part + right_part)
            
        return child

    def run(self):
        """
        Główny silnik (pętla) algorytmu ewolucyjnego.
        Odpowiada za inicjalizację, ocenę, sortowanie populacji, selekcję
        najlepszych (elityzm) oraz tworzenie nowych pokoleń w zadanej liczbie cykli.
        """
        population = self.initialize_population()
        history = []
        
        for gen in range(self.generations):
            scored_pop = [(self.calculate_sp_score(ind), ind) for ind in population]
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            
            best_score = scored_pop[0][0]
            history.append(best_score)
            
            if gen % 10 == 0 or gen == self.generations - 1:
                print(f"Generacja {gen}: Najlepszy SP-score = {best_score}")
            
            new_population = [scored_pop[0][1], scored_pop[1][1]]
            
            while len(new_population) < self.pop_size:
                p1 = random.choice(scored_pop[:self.pop_size//2])[1]
                
                if random.random() < self.crossover_rate:
                    p2 = random.choice(scored_pop[:self.pop_size//2])[1]
                    child = self.crossover(p1, p2)
                else:
                    child = list(p1)
                
                child = self.mutate(child)
                child = self.remove_empty_columns(child)
                new_population.append(child)
                
            population = new_population
            
        return population[0], history

if __name__ == "__main__":
    # Sekwencje białkowe z kilkoma aminokwasami z przykładu ze strony 4 instrukcji
    test_seqs = ["AGTCGTAGNPST", "ASTCGTAGPST", "GTPGAGNST", "ANCGTNPT"]
    ga = GeneticAlgorithmMSA(test_seqs, pop_size=30, generations=50, mutation_rate=0.3)
    best_alignment, _ = ga.run()
    
    print("\nOstateczne dopasowanie sekwencji białkowych (BLOSUM62):")
    for seq in best_alignment:
        print(seq)