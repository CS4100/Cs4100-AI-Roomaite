"""
filename: utils.py
Utility functions used for first choice hill climbing
room assignment system
Loads CSVs, builds arrays, scoring and output functions
"""

import csv
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_students_csv(file_path):
    """
    Load student data from a CSV file into a dictionary.

    Student ID for each row, (sleep, clean, noise) are integers,
    and roommate_preference with 'wants_' has a val of 1.
    """
    students = {}
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_id = int(row["student_id"])

            # Collect features student wants
            requested_features = [
                col for col, val in row.items()
                if col.startswith("wants_") and int(val) == 1
            ]
            students[s_id] = {
                "name": row["name"],
                "sleep": int(row["sleep"]), # sleep schedule pref (1-10)
                "clean": int(row["clean"]), # cleanliness pref (1-10)
                "noise": int(row["noise"]), # noise tolerance (1-10)
                "roommate_preference": int(row["roommate_count"]), #  preferred num of roommates
                "room_features": requested_features, # wanted features
            }
    return students

def load_rooms_csv(file_path):
    """
    Load rooms data from a CSV file into a dictionary.

    Room Id for each room, available features with 'has_'.
    """
    rooms = {}
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r_id = int(row["room_id"])

            # Features room has
            features = [
                col for col, val in row.items()
                if col.startswith("has_") and int(val) == 1
            ]
            rooms[r_id] = {
                "name": row["room_name"],
                "capacity": int(row["capacity"]), # Max num of students
                "features": features, # list of available features
            }
    return rooms


def build_arrays(students, rooms):
    """
    Build arrays for students and rooms.
    """
    # Extract ordered lists of IDs for indexing
    s_ids = list(students.keys())
    r_ids = list(rooms.keys())
    n_s = len(s_ids)
    n_r = len(r_ids)

    # ID to array index
    s_idx = {sid: i for i, sid in enumerate(s_ids)}
    r_idx = {rid: i for i, rid in enumerate(r_ids)}

    # Arrays of preferences
    sleep = [students[sid]["sleep"] for sid in s_ids]
    clean = [students[sid]["clean"] for sid in s_ids]
    noise = [students[sid]["noise"] for sid in s_ids]
    rpref = [students[sid]["roommate_preference"] for sid in s_ids]

    # Room capacities indexed by room array position
    cap = [rooms[rid]["capacity"] for rid in r_ids]

    # Compatability matrix: total preference mismatch between student i and j
    compat = [[0] * n_s for _ in range(n_s)]
    for i in range(n_s):
        for j in range(i + 1, n_s):
            d = abs(sleep[i] - sleep[j]) + abs(clean[i] - clean[j]) + abs(noise[i] - noise[j])
            # Matrix is symmetric
            compat[i][j] = d
            compat[j][i] = d

    # Feature cost matrix: number of features student i wants that room doesn't have
    feature_cost = [[0] * n_r for _ in range(n_s)]
    for i, sid in enumerate(s_ids):
        # Get the raw name (without wants_ or has_)
        wanted = {
            f[len("wants_"):] for f in students[sid]["room_features"]
            if f.startswith("wants_")
        }
        for j, rid in enumerate(r_ids):
            has = {
                f[len("has_"):] for f in rooms[rid]["features"]
                if f.startswith("has_")
            }

            # Count features the student wants and room doesn't have
            feature_cost[i][j] = len(wanted - has)

    return (s_ids, r_ids, s_idx, r_idx,
            sleep, clean, noise, rpref, cap,
            compat, feature_cost)


def random_assignment_arrays(n_s, cap, show_progress=True):
    """
    Generate a random valid initial assignment of students to rooms.
    """
    # Initialize empty room lists
    state = [[] for _ in range(len(cap))]
    s_indices = list(range(n_s))
    random.shuffle(s_indices)
    n_r = len(cap)

    it = s_indices
    if show_progress:
        it = tqdm(s_indices, desc="Creating initial assignment", unit="student")

    for si in it:
        placed = False
        # Keep trying random rooms until one is found with available capacity
        while not placed:
            ri = random.randrange(n_r)
            if len(state[ri]) < cap[ri]:
                state[ri].append(si)
                placed = True
    return state

def calculate_value_arrays(state, rpref, compat, feature_cost):
    """
    Compute the total objective cost of a given assignment using the pre-built arrays.
    lower score = better assignment
    """
    score = 0
    for ri, occ in enumerate(state):
        n = len(occ)
        # Preference mismatch between all roommates
        for a in range(n):
            for b in range(a + 1, n):
                score += compat[occ[a]][occ[b]]
        # Roommate count preference penalty
        actual = n - 1
        for si in occ:
            score += abs(rpref[si] - actual)
        # Feature mismatch penalty
        for si in occ:
            score += feature_cost[si][ri]
    return score

def calculate_value(state, students, rooms):
    """
    Compute the total objective cost.
    """
    # Initialize score
    score = 0
    # Preference mismatch
    for room, occupants in state.items():
        for i in range(len(occupants)):
            for j in range(i + 1, len(occupants)):
                s1 = students[occupants[i]]
                s2 = students[occupants[j]]
                score += abs(s1["sleep"] - s2["sleep"])
                score += abs(s1["clean"] - s2["clean"])
                score += abs(s1["noise"] - s2["noise"])

    # Roommate count preference penalty
    for room, occupants in state.items():
        actual = len(occupants) - 1
        for sid in occupants:
            score += abs(students[sid]["roommate_preference"] - actual)
    # Feature mismatch penalty
    for room, occupants in state.items():
        for sid in occupants:
            s = students[sid]
            wanted = {f[len("wants_"):] for f in s["room_features"] if f.startswith("wants_")}
            has = {f[len("has_"):] for f in rooms[room]["features"] if f.startswith("has_")}
            score += len(wanted - has)
    return score

def sample_room_assignments(assignment, students, rooms, n=4):
    """
    Display sample room assignments
    I used AI here to assist me in printing out the sample room assignment. I needed help with the process of this
    to ensure I printed it out correctly!
    """
    results = []
    for room_id, occupants in assignment.items():
        if not occupants:
            continue
        cost = 0
        # preference mismatch
        for i in range(len(occupants)):
            for j in range(i + 1, len(occupants)):
                s1, s2 = students[occupants[i]], students[occupants[j]]
                cost += abs(s1["sleep"]-s2["sleep"]) + abs(s1["clean"]-s2["clean"]) + abs(s1["noise"]-s2["noise"])
        # roommate count penalty
        actual = len(occupants) - 1
        for sid in occupants:
            cost += abs(students[sid]["roommate_preference"] - actual)
        # feature mismatch
        for sid in occupants:
            wanted = {f[6:] for f in students[sid]["room_features"] if f.startswith("wants_")}
            has = {f[4:] for f in rooms[room_id]["features"] if f.startswith("has_")}
            cost += len(wanted - has)
        results.append((room_id, occupants, cost))

    results = [r for r in results if len(r[1]) > 1]
    results.sort(key=lambda x: x[2])
    for room_id, occupants, cost in results[:n]:
        names = ", ".join(f"Student_{sid}" for sid in occupants)
        print(f"Room_{room_id} | {len(occupants)} | {names} | {cost}")

def plot_history(history, n_students, n_rooms, elapsed, output_path):
    """
    Plot and save the score history over hill climbing iterations.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Total Cost")
    plt.title(f"First Choice Hill Climbing - {n_students} students, {n_rooms} rooms ({elapsed:.1f}s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved figure to {output_path}")

def save_results(output_path, assignment, students, rooms):
    """
    Write the final room assignment and score to a text file.
    """
    final_score = calculate_value(assignment, students, rooms)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("First Choice Hill Climbing Results\n")
        f.write(f"Final score: {final_score}\n\n")
        # Write each room sorted by room_id for consistent output
        for room_id in sorted(assignment.keys()):
            room = rooms[room_id]
            occupants = assignment[room_id]
            f.write(f"Room {room_id} ({room['name']}) - {len(occupants)}/{room['capacity']} occupants\n")
            for sid in occupants:
                f.write(f" - {sid}: {students[sid]['name']}\n")
            if not occupants:
                f.write(" - <empty>\n")
            f.write("\n")