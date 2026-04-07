"""
simulated_annealing.py - SA for roommate matching
CS4100 AI Project - Shray

assigns students to rooms  considering both:
-how compatible roommates are with each other
-how well the room features match what students want
"""

import random
import math
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.objective import (
    load_data, assignment_cost, student_pair_cost, room_match_cost
)


def make_initial_assignment(n_students, room_capacities):
    """
    randomly assign students to rooms
    picks rooms until we have enough capacity for all students
    """
    # figures out which rooms we need
    indices = list(range(len(room_capacities)))
    random.shuffle(indices)

    selected_rooms = []
    total_cap = 0
    for idx in indices:
        selected_rooms.append(idx)
        total_cap += room_capacities[idx]
        if total_cap >= n_students:
            break

    # shuffles students and fill rooms
    students = list(range(n_students))
    random.shuffle(students)

    assignment = {}
    pos = 0
    for room_idx in selected_rooms:
        cap = room_capacities[room_idx]
        end = min(pos + cap, n_students)
        if pos < end:
            assignment[room_idx] = students[pos:end]
            pos = end
        if pos >= n_students:
            break

    return assignment


def get_neighbor(assignment):
    """
    swap one student between two random rooms
    """
    rooms = list(assignment.keys())
    if len(rooms) < 2:
        return assignment

    r1, r2 = random.sample(rooms, 2)

    if len(assignment[r1]) == 0 or len(assignment[r2]) == 0:
        return assignment

    # picks a random student from each room and swap
    i = random.randint(0, len(assignment[r1]) - 1)
    j = random.randint(0, len(assignment[r2]) - 1)

    new_assignment = {k: v[:] for k, v in assignment.items()}
    new_assignment[r1][i], new_assignment[r2][j] = new_assignment[r2][j], new_assignment[r1][i]

    return new_assignment


def room_cost(room_idx, student_indices, student_prefs, student_features, room_features):
    """cost for a single room - used for fast delta calculation"""
    cost = 0
    for i in range(len(student_indices)):
        for j in range(i + 1, len(student_indices)):
            cost += student_pair_cost(
                student_prefs, student_features,
                student_indices[i], student_indices[j]
            )
        cost += room_match_cost(
            student_features, room_features,
            student_indices[i], room_idx
        )
    return cost


def simulated_annealing(
    student_prefs, student_features, room_features, room_capacities,
    initial_temp=100.0, cooling_rate=0.9999, min_temp=0.01, max_iter=50000
):
    """
    SA that assigns students to rooms
    only recalculates cost for the two rooms that changed
    """
    n_students = len(student_prefs)

    current = make_initial_assignment(n_students, room_capacities)
    current_cost = assignment_cost(current, student_prefs, student_features, room_features)

    best = {k: v[:] for k, v in current.items()}
    best_cost = current_cost

    temp = initial_temp
    cost_history = [current_cost]

    start_time = time.time()

    for iteration in range(max_iter):
        if temp < min_temp:
            break

        # pick two rooms to swap between
        rooms = list(current.keys())
        if len(rooms) < 2:
            break

        r1, r2 = random.sample(rooms, 2)
        if len(current[r1]) == 0 or len(current[r2]) == 0:
            continue

        # old cost of just these two rooms
        old_cost_r1 = room_cost(r1, current[r1], student_prefs, student_features, room_features)
        old_cost_r2 = room_cost(r2, current[r2], student_prefs, student_features, room_features)

        # make the swap
        i = random.randint(0, len(current[r1]) - 1)
        j = random.randint(0, len(current[r2]) - 1)

        new_r1 = current[r1][:]
        new_r2 = current[r2][:]
        new_r1[i], new_r2[j] = new_r2[j], new_r1[i]

        # new cost of just these two rooms
        new_cost_r1 = room_cost(r1, new_r1, student_prefs, student_features, room_features)
        new_cost_r2 = room_cost(r2, new_r2, student_prefs, student_features, room_features)

        delta = (new_cost_r1 + new_cost_r2) - (old_cost_r1 + old_cost_r2)

        if delta < 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta / temp)

        if accept:
            current[r1] = new_r1
            current[r2] = new_r2
            current_cost += delta

        if current_cost < best_cost:
            best = {k: v[:] for k, v in current.items()}
            best_cost = current_cost

        cost_history.append(current_cost)
        temp *= cooling_rate

    elapsed = time.time() - start_time
    return best, best_cost, cost_history, elapsed


# running it
if __name__ == "__main__":
    random.seed(42)

    students_df, rooms_df, student_prefs, student_features, room_features, room_capacities = \
        load_data("data/students.csv", "data/rooms.csv")

    print(f"running SA on {len(students_df)} students, {len(rooms_df)} rooms...\n")

    best_assignment, best_cost, history, elapsed = simulated_annealing(
        student_prefs, student_features, room_features, room_capacities,
        initial_temp=100.0,
        cooling_rate=0.9999,
        min_temp=0.01,
        max_iter=50000
    )

    print(f"starting cost:  {history[0]}")
    print(f"final cost:     {best_cost}")
    print(f"iterations:     {len(history)}")
    print(f"runtime:        {elapsed:.2f} seconds")
    print(f"rooms used:     {len(best_assignment)}\n")

    # show first 10 rooms
    print("sample room assignments (first 10):")
    for idx, (room_idx, student_indices) in enumerate(list(best_assignment.items())[:10]):
        room_name = rooms_df.iloc[room_idx]["room_name"]
        cap = room_capacities[room_idx]
        student_names = [students_df.iloc[s]["name"] for s in student_indices]
        rc = room_cost(room_idx, student_indices, student_prefs, student_features, room_features)
        print(f"  {room_name} (cap {cap}): {', '.join(student_names)}  | cost: {rc}")

    # plot
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(history, linewidth=0.5, color="steelblue")
        plt.xlabel("iteration")
        plt.ylabel("total cost")
        plt.title(f"simulated annealing - {len(students_df)} students, {len(best_assignment)} rooms ({elapsed:.1f}s)")
        plt.tight_layout()
        plt.savefig("results/sa_cost_history.png", dpi=150)
        print("\nplot saved to results/sa_cost_history.png")
    except ImportError:
        print("\n(install matplotlib to see the plot)")