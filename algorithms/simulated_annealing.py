"""
simulated_annealing.py - SA for roommate matching
CS4100 AI Project - Shray
"""

import random
import math
import copy
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.objective import Student, pair_cost, total_cost
from shared.data import generate_students


def make_initial_pairs(students):
    """randomly pair up students as a starting point"""
    shuffled = students[:]
    random.shuffle(shuffled)
    pairs = []
    for i in range(0, len(shuffled) - 1, 2):
        pairs.append((shuffled[i], shuffled[i + 1]))
    return pairs


def get_neighbor(pairs):
    """
    should take the current pairing and make a small change 
    and hopefully swap one person from one pair with someone from another pair
    """
    new_pairs = copy.deepcopy(pairs)
    if len(new_pairs) < 2:
        return new_pairs

    i, j = random.sample(range(len(new_pairs)), 2)
    pos_i = random.randint(0, 1)
    pos_j = random.randint(0, 1)

    pair_i = list(new_pairs[i])
    pair_j = list(new_pairs[j])
    pair_i[pos_i], pair_j[pos_j] = pair_j[pos_j], pair_i[pos_i]
    new_pairs[i] = tuple(pair_i)
    new_pairs[j] = tuple(pair_j)

    return new_pairs


def simulated_annealing(students, initial_temp=100.0, cooling_rate=0.995, min_temp=0.01, max_iter=10000):
    """
    what we gotta do
    - start with random pairs
    - generate neighbor by swapping
    - accept better solutions always, worse ones sometimes (based on temp)
    - cool down over time so we accept less bad moves as we go
    """
    current = make_initial_pairs(students)
    current_cost = total_cost(current)

    best = copy.deepcopy(current)
    best_cost = current_cost

    temp = initial_temp

    cost_history = [current_cost]

    for i in range(max_iter):
        if temp < min_temp:
            break

        neighbor = get_neighbor(current)
        neighbor_cost = total_cost(neighbor)

        delta = neighbor_cost - current_cost

        # always accept if better, sometimes accept if worse
        if delta < 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta / temp)

        if accept:
            current = neighbor
            current_cost = neighbor_cost

        if current_cost < best_cost:
            best = copy.deepcopy(current)
            best_cost = current_cost

        temp *= cooling_rate
        cost_history.append(current_cost)

    return best, best_cost, cost_history

# --- run it ---
if __name__ == "__main__":
    random.seed(42)

    # generate test students (no LLM, just random)
    students = generate_students(20)
    print(f"running SA on {len(students)} students...\n")

    best_pairs, best_cost, history = simulated_annealing(students)

    print(f"starting cost: {history[0]}")
    print(f"final cost:    {best_cost}")
    print(f"iterations:    {len(history)}\n")

    print("best pairings:")
    for a, b in best_pairs:
        c = pair_cost(a, b)
        print(f"  {a.name:>12} <-> {b.name:<12}  cost: {c}")