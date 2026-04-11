"""
simulated_annealing.py - sa for roommate matching
cs4100 ai project - shray

assigns students to rooms using the same objective function
as the hill climbing so we can compare results fairly
"""

import random
import math
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.objective import (
    load_data, assignment_cost, room_cost
)


def make_initial_assignment(n_students, room_capacities):
    """randomly throw students into rooms as a starting point"""
    indices = list(range(len(room_capacities)))
    random.shuffle(indices)

    # pick rooms until we have enough space for everyone
    selected_rooms = []
    total_cap = 0
    for idx in indices:
        selected_rooms.append(idx)
        total_cap += room_capacities[idx]
        if total_cap >= n_students:
            break

    # shuffle students and fill rooms up
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


def simulated_annealing(
    student_prefs, roommate_pref, student_features, room_features, room_capacities,
    initial_temp=100.0, cooling_rate=0.9999, min_temp=0.01, max_iter=50000
):
    """
    main sa loop
    - start with random room assignments
    - each iteration pick two rooms and swap a student between them
    - always accept if its better, sometimes accept if worse (based on temp)
    - cool down so we stop accepting bad moves over time
    - only recalculate cost for the two rooms that changed not all of them
    """
    n_students = len(student_prefs)

    current = make_initial_assignment(n_students, room_capacities)
    current_cost = assignment_cost(current, student_prefs, roommate_pref,
                                   student_features, room_features)

    best = {k: v[:] for k, v in current.items()}
    best_cost = current_cost

    temp = initial_temp
    cost_history = [current_cost]

    start_time = time.time()

    for iteration in range(max_iter):
        if temp < min_temp:
            break

        rooms = list(current.keys())
        if len(rooms) < 2:
            break

        # pick two random rooms
        r1, r2 = random.sample(rooms, 2)
        if len(current[r1]) == 0 or len(current[r2]) == 0:
            continue

        # cost of these two rooms before the swap
        old_cost_r1 = room_cost(r1, current[r1], student_prefs, roommate_pref,
                                student_features, room_features)
        old_cost_r2 = room_cost(r2, current[r2], student_prefs, roommate_pref,
                                student_features, room_features)

        # pick one student from each room and swap them
        i = random.randint(0, len(current[r1]) - 1)
        j = random.randint(0, len(current[r2]) - 1)

        new_r1 = current[r1][:]
        new_r2 = current[r2][:]
        new_r1[i], new_r2[j] = new_r2[j], new_r1[i]

        # cost after the swap
        new_cost_r1 = room_cost(r1, new_r1, student_prefs, roommate_pref,
                                student_features, room_features)
        new_cost_r2 = room_cost(r2, new_r2, student_prefs, roommate_pref,
                                student_features, room_features)

        delta = (new_cost_r1 + new_cost_r2) - (old_cost_r1 + old_cost_r2)

        # always accept better, sometimes accept worse
        if delta < 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta / temp)

        if accept:
            current[r1] = new_r1
            current[r2] = new_r2
            current_cost += delta

        # keep track of the best we ever found
        if current_cost < best_cost:
            best = {k: v[:] for k, v in current.items()}
            best_cost = current_cost

        cost_history.append(current_cost)
        temp *= cooling_rate

    elapsed = time.time() - start_time
    return best, best_cost, cost_history, elapsed


# --- run it ---
if __name__ == "__main__":
    random.seed(42)

    # load the data
    students_df, rooms_df, student_prefs, roommate_pref, \
        student_features, room_features, room_capacities = \
        load_data("data/students.csv", "data/rooms.csv")

    print(f"running sa on {len(students_df)} students, {len(rooms_df)} rooms...\n")

    best_assignment, best_cost, history, elapsed = simulated_annealing(
        student_prefs, roommate_pref, student_features, room_features, room_capacities,
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

    # show some example rooms
    print("sample room assignments (first 10):")
    for idx, (room_idx, student_indices) in enumerate(list(best_assignment.items())[:10]):
        room_name = rooms_df.iloc[room_idx]["room_name"]
        cap = room_capacities[room_idx]
        student_names = [students_df.iloc[s]["name"] for s in student_indices]
        rc = room_cost(room_idx, student_indices, student_prefs, roommate_pref,
                       student_features, room_features)
        print(f"  {room_name} (cap {cap}): {', '.join(student_names)}  | cost: {rc}")

    # plot the cost over time
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