import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

def load_students_csv(file_path):
    students = {}
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["student_id"])
            requested = [
                col for col, val in row.items()
                if col.startswith("wants_") and int(val) == 1
            ]
            students[sid] = {
                "name": row["name"],
                "sleep": int(row["sleep"]),
                "clean": int(row["clean"]),
                "noise": int(row["noise"]),
                "roommate_preference": int(row["roommate_count"]),
                "room_features": requested,
            }
    return students


def load_rooms_csv(file_path):
    rooms = {}
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = int(row["room_id"])
            features = [
                col for col, val in row.items()
                if col.startswith("has_") and int(val) == 1
            ]
            rooms[rid] = {
                "name": row["room_name"],
                "capacity": int(row["capacity"]),
                "features": features,
            }
    return rooms

def build_arrays(students, rooms):
    s_ids = list(students.keys())
    r_ids = list(rooms.keys())
    n_s = len(s_ids)
    n_r = len(r_ids)

    s_idx = {sid: i for i, sid in enumerate(s_ids)}
    r_idx = {rid: i for i, rid in enumerate(r_ids)}

    sleep = [students[sid]["sleep"] for sid in s_ids]
    clean = [students[sid]["clean"] for sid in s_ids]
    noise = [students[sid]["noise"] for sid in s_ids]
    rpref = [students[sid]["roommate_preference"] for sid in s_ids]
    cap = [rooms[rid]["capacity"] for rid in r_ids]

    compat = [[0] * n_s for _ in range(n_s)]
    for i in range(n_s):
        for j in range(i + 1, n_s):
            d = abs(sleep[i] - sleep[j]) + abs(clean[i] - clean[j]) + abs(noise[i] - noise[j])
            compat[i][j] = d
            compat[j][i] = d

    feature_cost = [[0] * n_r for _ in range(n_s)]
    for i, sid in enumerate(s_ids):
        wanted = {
            f[len("wants_"):] for f in students[sid]["room_features"]
            if f.startswith("wants_")
        }
        for j, rid in enumerate(r_ids):
            has = {
                f[len("has_"):] for f in rooms[rid]["features"]
                if f.startswith("has_")
            }
            feature_cost[i][j] = len(wanted - has)

    return (s_ids, r_ids, s_idx, r_idx,
            sleep, clean, noise, rpref, cap,
            compat, feature_cost)

def random_assignment_arrays(n_s, cap, show_progress=True):
    state = [[] for _ in range(len(cap))]
    s_indices = list(range(n_s))
    random.shuffle(s_indices)
    n_r = len(cap)

    it = s_indices
    if show_progress and TQDM_AVAILABLE:
        it = tqdm(s_indices, desc="Creating initial assignment", unit="student")

    for si in it:
        placed = False
        while not placed:
            ri = random.randrange(n_r)
            if len(state[ri]) < cap[ri]:
                state[ri].append(si)
                placed = True
    return state

def calculate_value_arrays(state, rpref, compat, feature_cost):
    score = 0
    for ri, occ in enumerate(state):
        n = len(occ)
        for a in range(n):
            for b in range(a + 1, n):
                score += compat[occ[a]][occ[b]]
        actual = n - 1
        for si in occ:
            score += abs(rpref[si] - actual)
        for si in occ:
            score += feature_cost[si][ri]
    return score

def first_choice_hill_climbing(students, rooms, show_progress=True, max_iterations=1000):
    (s_ids, r_ids, s_idx, r_idx,
     sleep, clean, noise, rpref, cap,
     compat, feature_cost) = build_arrays(students, rooms)

    n_r = len(r_ids)
    state = random_assignment_arrays(len(s_ids), cap, show_progress=show_progress)
    current_value = calculate_value_arrays(state, rpref, compat, feature_cost)
    history = [current_value]

    for iteration in range(1, max_iterations + 1):
        if show_progress:
            print(f"Iteration {iteration}: score={current_value}")

        improved = False

        room_indices = list(range(n_r))
        random.shuffle(room_indices)

        for ri in room_indices:
            if improved:
                break
            occ1 = state[ri]
            if not occ1:
                continue

            other_rooms = list(range(n_r))
            random.shuffle(other_rooms)

            for rj in other_rooms:
                if improved:
                    break
                if rj == ri:
                    continue
                occ2 = state[rj]
                if not occ2:
                    continue

                actual1 = len(occ1) - 1
                actual2 = len(occ2) - 1

                candidates1 = list(occ1)
                candidates2 = list(occ2)
                random.shuffle(candidates1)
                random.shuffle(candidates2)

                for s1 in candidates1:
                    if improved:
                        break

                    old_lifestyle_s1 = sum(compat[s1][o] for o in occ1 if o != s1)
                    old_rpref_s1 = abs(rpref[s1] - actual1)
                    old_feat_s1 = feature_cost[s1][ri]

                    for s2 in candidates2:
                        old_lifestyle_s2 = sum(compat[s2][o] for o in occ2 if o != s2)
                        old_rpref_s2 = abs(rpref[s2] - actual2)
                        old_feat_s2 = feature_cost[s2][rj]

                        old = (old_lifestyle_s1 + old_rpref_s1 + old_feat_s1
                               + old_lifestyle_s2 + old_rpref_s2 + old_feat_s2)

                        new_lifestyle_s2_in_ri = sum(compat[s2][o] for o in occ1 if o != s1)
                        new_rpref_s2 = abs(rpref[s2] - actual1)
                        new_feat_s2 = feature_cost[s2][ri]

                        new_lifestyle_s1_in_rj = sum(compat[s1][o] for o in occ2 if o != s2)
                        new_rpref_s1 = abs(rpref[s1] - actual2)
                        new_feat_s1 = feature_cost[s1][rj]

                        new = (new_lifestyle_s2_in_ri + new_rpref_s2 + new_feat_s2
                               + new_lifestyle_s1_in_rj + new_rpref_s1 + new_feat_s1)

                        delta = new - old

                        if delta < 0:
                            state[ri].remove(s1)
                            state[ri].append(s2)
                            state[rj].remove(s2)
                            state[rj].append(s1)
                            current_value += delta
                            history.append(current_value)
                            improved = True
                            if show_progress:
                                print(f" → Accepted swap, new score={current_value}")
                            break

        if not improved:
            if show_progress:
                print(f"Stopped at iteration {iteration}: local optimum reached.")
            break

    else:
        if show_progress:
            print(f"Stopped after max_iterations={max_iterations}.")

    assignment = {}
    for ri, occ in enumerate(state):
        assignment[r_ids[ri]] = [s_ids[si] for si in occ]
    return assignment, history

def calculate_value(state, students, rooms):
    score = 0
    for room, occupants in state.items():
        for i in range(len(occupants)):
            for j in range(i + 1, len(occupants)):
                s1 = students[occupants[i]]
                s2 = students[occupants[j]]
                score += abs(s1["sleep"] - s2["sleep"])
                score += abs(s1["clean"] - s2["clean"])
                score += abs(s1["noise"] - s2["noise"])
    for room, occupants in state.items():
        actual = len(occupants) - 1
        for sid in occupants:
            score += abs(students[sid]["roommate_preference"] - actual)
    for room, occupants in state.items():
        for sid in occupants:
            s = students[sid]
            wanted = {f[len("wants_"):] for f in s["room_features"] if f.startswith("wants_")}
            has    = {f[len("has_"):] for f in rooms[room]["features"] if f.startswith("has_")}
            score += len(wanted - has)
    return score

def plot_history(history, n_students, n_rooms, elapsed, output_path):
    plt.figure(figsize=(12, 5))
    plt.plot(history)
    plt.xlabel("iteration")
    plt.ylabel("total cost")
    plt.title(f"first choice hill climbing - {n_students} students, {n_rooms} rooms ({elapsed:.1f}s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved figure to {output_path}")

def save_results(output_path, assignment, students, rooms):
    final_score = calculate_value(assignment, students, rooms)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("First Choice Hill Climbing Results\n")
        f.write(f"Final score: {final_score}\n\n")
        for room_id in sorted(assignment.keys()):
            room = rooms[room_id]
            occupants = assignment[room_id]
            f.write(f"Room {room_id} ({room['name']}) - {len(occupants)}/{room['capacity']} occupants\n")
            for sid in occupants:
                f.write(f" - {sid}: {students[sid]['name']}\n")
            if not occupants:
                f.write(" - <empty>\n")
            f.write("\n")

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    output_path = root / "results" / "output.txt"
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "first_choice_hill_climbing.png"

    print("Loading input CSV files...")
    students = load_students_csv(data_dir / "students.csv")
    rooms = load_rooms_csv(data_dir / "rooms.csv")
    print(f"Loaded {len(students)} students and {len(rooms)} rooms.")

    if not TQDM_AVAILABLE:
        print("tqdm not installed — basic progress only. pip install tqdm")

    start = time.time()
    assignment, history = first_choice_hill_climbing(
        students, rooms, show_progress=True, max_iterations=1000)
    elapsed = time.time() - start

    best_score = calculate_value(assignment, students, rooms)
    print(f"Final score: {best_score}")
    save_results(output_path, assignment, students, rooms)
    print(f"Saved results to {output_path}")

    plot_history(history, len(students), len(rooms), elapsed, results_path)

if __name__ == "__main__":
    main()