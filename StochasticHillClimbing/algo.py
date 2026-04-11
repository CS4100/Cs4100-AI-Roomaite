"""
algo.py - Stochastic Hill Climbing for roommate matching
CS4100 AI Project

Objective function:
score =
    sum of pairwise roommate incompatibility
    + sum of roommate-count mismatch penalties
    + sum of room-feature mismatch penalties
"""

import random
import time
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREF_COLS = ["sleep", "clean", "noise"]
ROOMMATE_COUNT_COL = "roommate_count"

FEATURE_NAMES = [
    "ac",
    "private_bath",
    "balcony",
    "kitchen",
    "laundry",
    "wifi",
    "parking",
]

STUDENT_FEATURE_COLS = [f"wants_{f}" for f in FEATURE_NAMES]
ROOM_FEATURE_COLS = [f"has_{f}" for f in FEATURE_NAMES]


# Load data
def load_data(students_path, rooms_path):
    """
    Load student and room CSV files and return the dataframes
    plus numpy arrays for fast scoring.
    """
    students_df = pd.read_csv(students_path)
    rooms_df = pd.read_csv(rooms_path)

    student_prefs = students_df[PREF_COLS].values.astype(int)
    student_roommate_counts = students_df[ROOMMATE_COUNT_COL].values.astype(int)
    student_features = students_df[STUDENT_FEATURE_COLS].values.astype(int)

    room_features = rooms_df[ROOM_FEATURE_COLS].values.astype(int)
    room_capacities = rooms_df["capacity"].values.astype(int)

    return (
        students_df,
        rooms_df,
        student_prefs,
        student_roommate_counts,
        student_features,
        room_features,
        room_capacities,
    )


def student_pair_cost(student_prefs, i, j):
    """
    Pairwise incompatibility between two students:
    sum of absolute differences in sleep, clean, and noise.
    """
    return int(np.sum(np.abs(student_prefs[i] - student_prefs[j])))


def roommate_count_penalty(student_roommate_counts, student_idx, room_size):
    """
    Penalty for mismatch between preferred roommate count
    and actual roommate count.

    actual roommate count = room size - 1
    """
    actual_roommates = room_size - 1
    preferred_roommates = int(student_roommate_counts[student_idx])
    return abs(preferred_roommates - actual_roommates)


def room_match_cost(student_features, room_features, student_idx, room_idx):
    """
    Add 1 for each wanted feature that the room does not have.
    """
    wants = student_features[student_idx]
    has = room_features[room_idx]
    return int(np.sum(wants * (1 - has)))


def room_cost(
    room_idx,
    student_indices,
    student_prefs,
    student_roommate_counts,
    student_features,
    room_features,
):
    """
    Cost for one room:
    1. pairwise incompatibility
    2. roommate-count mismatch penalty
    3. room-feature mismatch penalty
    """
    total = 0
    room_size = len(student_indices)

    for i in range(room_size):
        for j in range(i + 1, room_size):
            total += student_pair_cost(
                student_prefs,
                student_indices[i],
                student_indices[j],
            )

        total += roommate_count_penalty(
            student_roommate_counts,
            student_indices[i],
            room_size,
        )

        total += room_match_cost(
            student_features,
            room_features,
            student_indices[i],
            room_idx,
        )

    return total


def assignment_cost(
    assignment,
    student_prefs,
    student_roommate_counts,
    student_features,
    room_features,
):
    """
    Total cost of an assignment.
    assignment = {room_idx: [student_idx, ...]}
    """
    total = 0

    for room_idx, student_indices in assignment.items():
        total += room_cost(
            room_idx,
            student_indices,
            student_prefs,
            student_roommate_counts,
            student_features,
            room_features,
        )

    return total

# Stochastic Hill Climbing algorithm
def make_initial_assignment(n_students, room_capacities):
    """
    Randomly assign students to rooms.

    Randomly shuffles room indices, picks rooms until total capacity
    is enough for all students, then fills them with shuffled students.
    """
    indices = list(range(len(room_capacities)))
    random.shuffle(indices)

    selected_rooms = []
    total_cap = 0
    for idx in indices:
        selected_rooms.append(idx)
        total_cap += room_capacities[idx]
        if total_cap >= n_students:
            break

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


def stochastic_hill_climbing(
    student_prefs,
    student_roommate_counts,
    student_features,
    room_features,
    room_capacities,
    max_iter=5000,
):
    """
    Stochastic hill climbing:
    - start from a random valid assignment
    - each iteration generates one random neighbor
    - neighbor = swap one random student between two random rooms
    - accept only if the score improves
    """
    n_students = len(student_prefs)

    current = make_initial_assignment(n_students, room_capacities)
    current_cost = assignment_cost(
        current,
        student_prefs,
        student_roommate_counts,
        student_features,
        room_features,
    )

    best = {k: v[:] for k, v in current.items()}
    best_cost = current_cost
    history = [current_cost]

    start_time = time.time()

    for _ in range(max_iter):
        rooms = list(current.keys())
        if len(rooms) < 2:
            break

        r1, r2 = random.sample(rooms, 2)

        if len(current[r1]) == 0 or len(current[r2]) == 0:
            history.append(best_cost)
            continue

        old_cost_r1 = room_cost(
            r1,
            current[r1],
            student_prefs,
            student_roommate_counts,
            student_features,
            room_features,
        )
        old_cost_r2 = room_cost(
            r2,
            current[r2],
            student_prefs,
            student_roommate_counts,
            student_features,
            room_features,
        )

        i = random.randint(0, len(current[r1]) - 1)
        j = random.randint(0, len(current[r2]) - 1)

        new_r1 = current[r1][:]
        new_r2 = current[r2][:]

        new_r1[i], new_r2[j] = new_r2[j], new_r1[i]

        new_cost_r1 = room_cost(
            r1,
            new_r1,
            student_prefs,
            student_roommate_counts,
            student_features,
            room_features,
        )
        new_cost_r2 = room_cost(
            r2,
            new_r2,
            student_prefs,
            student_roommate_counts,
            student_features,
            room_features,
        )

        delta = (new_cost_r1 + new_cost_r2) - (old_cost_r1 + old_cost_r2)

        # only accept better moves
        if delta < 0:
            current[r1] = new_r1
            current[r2] = new_r2
            current_cost += delta

            if current_cost < best_cost:
                best = {k: v[:] for k, v in current.items()}
                best_cost = current_cost

        history.append(best_cost)

    elapsed = time.time() - start_time
    return best, best_cost, history, elapsed


if __name__ == "__main__":
    random.seed(42)

    base_dir = Path(__file__).resolve().parent
    students_path = base_dir.parent / "data" / "students.csv"
    rooms_path = base_dir.parent / "data" / "rooms.csv"

    (
        students_df,
        rooms_df,
        student_prefs,
        student_roommate_counts,
        student_features,
        room_features,
        room_capacities,
    ) = load_data(students_path, rooms_path)

    print(
        f"running stochastic hill climbing on "
        f"{len(students_df)} students, {len(rooms_df)} rooms...\n"
    )

    best_assignment, best_cost, history, elapsed = stochastic_hill_climbing(
        student_prefs,
        student_roommate_counts,
        student_features,
        room_features,
        room_capacities,
        max_iter=5000,
    )

    starting_cost = history[0]
    improvement = (
        (starting_cost - best_cost) / starting_cost * 100
        if starting_cost != 0 else 0.0
    )

    print(f"starting cost:  {starting_cost}")
    print(f"final cost:     {best_cost}")
    print(f"improvement:    {improvement:.1f}%")
    print(f"iterations:     {len(history)}")
    print(f"runtime:        {elapsed:.2f} seconds")
    print(f"rooms used:     {len(best_assignment)}\n")

# Print first 10 sample room assignments
    print("sample room assignments (first 10):")
    for room_idx, student_indices in list(best_assignment.items())[:10]:
        room_name = rooms_df.iloc[room_idx]["room_name"]
        cap = room_capacities[room_idx]
        student_names = [students_df.iloc[s]["name"] for s in student_indices]
        rc = room_cost(
            room_idx,
            student_indices,
            student_prefs,
            student_roommate_counts,
            student_features,
            room_features,
        )
        print(
            f"  {room_name} (cap {cap}): "
            f"{', '.join(student_names)} | cost: {rc}"
        )

    os.makedirs("results", exist_ok=True)

# Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history, linewidth=0.8)
    plt.xlabel("iteration")
    plt.ylabel("total cost")
    plt.title(
        f"stochastic hill climbing - {len(students_df)} students, "
        f"{len(best_assignment)} rooms ({elapsed:.1f}s)"
    )
    plt.tight_layout()
    plt.savefig("results/shc_cost_history.png", dpi=150)
    print("\nplot saved to results/shc_cost_history.png")