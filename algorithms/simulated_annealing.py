"""
simulated_annealing.py - SA for roommate matching
CS4100 AI Project - Shray

optimized for 10k students - only recalculates the 
cost of the two pairs that changed, not all 5000 pairs
"""

import random
import math
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.objective import load_student_matrix, pair_cost, total_cost


def make_initial_pairs(n):
    """randomly pair up student indices"""
    indices = list(range(n))
    random.shuffle(indices)
    pairs = []
    for i in range(0, len(indices) - 1, 2):
        pairs.append((indices[i], indices[i + 1]))
    return pairs


def simulated_annealing(matrix, initial_temp=100.0, cooling_rate=0.995, min_temp=0.01, max_iter=10000):
    """
    main SA loop - optimized so we only recalculate the 
    cost of the two swapped pairs instead of all pairs
    """
    n = len(matrix)

    current = make_initial_pairs(n)
    current_cost = total_cost(matrix, current)

    best = current[:]
    best_cost = current_cost

    temp = initial_temp
    cost_history = [current_cost]

    start_time = time.time()

    for iteration in range(max_iter):
        if temp < min_temp:
            break

        # pick two random pairs to swap between
        i, j = random.sample(range(len(current)), 2)
        pos_i = random.randint(0, 1)
        pos_j = random.randint(0, 1)

        # cost of old pairs
        old_cost_i = pair_cost(matrix, current[i][0], current[i][1])
        old_cost_j = pair_cost(matrix, current[j][0], current[j][1])

        # make the swap
        pair_i = list(current[i])
        pair_j = list(current[j])
        pair_i[pos_i], pair_j[pos_j] = pair_j[pos_j], pair_i[pos_i]

        # cost of new pairs
        new_cost_i = pair_cost(matrix, pair_i[0], pair_i[1])
        new_cost_j = pair_cost(matrix, pair_j[0], pair_j[1])

        # delta is just the change in these two pairs
        delta = (new_cost_i + new_cost_j) - (old_cost_i + old_cost_j)

        if delta < 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta / temp)

        if accept:
            current[i] = tuple(pair_i)
            current[j] = tuple(pair_j)
            current_cost += delta

        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost

        cost_history.append(current_cost)
        temp *= cooling_rate

    elapsed = time.time() - start_time

    return best, best_cost, cost_history, elapsed


#    run it 
if __name__ == "__main__":
    random.seed(42)

    # load student data
    df, matrix = load_student_matrix("data/students.csv")
    print(f"running SA on {len(df)} students...\n")

    best_pairs, best_cost, history, elapsed = simulated_annealing(
        matrix,
        initial_temp=100.0,
        cooling_rate=0.9999,
        max_iter=50000
    )

    print(f"starting cost:  {history[0]}")
    print(f"final cost:     {best_cost}")
    print(f"iterations:     {len(history)}")
    print(f"runtime:        {elapsed:.2f} seconds\n")

    # show first 10 pairs
    print("sample pairings (first 10):")
    for a, b in best_pairs[:10]:
        c = pair_cost(matrix, a, b)
        name_a = df.iloc[a]["name"]
        name_b = df.iloc[b]["name"]
        print(f"  {name_a:>14} <-> {name_b:<14}  cost: {c}")

    # plot
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(history, linewidth=0.5, color="steelblue")
        plt.xlabel("iteration")
        plt.ylabel("total cost")
        plt.title(f"simulated annealing - {len(df)} students ({elapsed:.1f}s)")
        plt.tight_layout()
        plt.savefig("results/sa_cost_history.png", dpi=150)
        print("\nplot saved to results/sa_cost_history.png")
    except ImportError:
        print("\n(install matplotlib to see the plot)")

