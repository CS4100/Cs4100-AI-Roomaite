import csv
import random
from pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os

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
    print("Building index arrays...")
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
            d = (abs(sleep[i] - sleep[j])
                 + abs(clean[i] - clean[j])
                 + abs(noise[i] - noise[j]))
            compat[i][j] = d
            compat[j][i] = d

    print(f"  Built compatibility matrix ({n_s}×{n_s})...")
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

    print(f"  Built feature cost matrix ({n_s}×{n_r})...")
    return (s_ids, r_ids, s_idx, r_idx,
            sleep, clean, noise, rpref, cap,
            compat, feature_cost)

def random_assignment_arrays(n_s, cap):
    print(f"Generating random initial assignment for {n_s} students into {len(cap)} rooms...")
    state = [[] for _ in range(len(cap))]
    s_indices = list(range(n_s))
    random.shuffle(s_indices)
    n_r = len(cap)
    for si in s_indices:
        placed = False
        while not placed:
            ri = random.randrange(n_r)
            if len(state[ri]) < cap[ri]:
                state[ri].append(si)
                placed = True
    print("  Initial assignment done.")
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

def _evaluate_chunk(args):
    room_pairs, state_snapshot, rpref, compat, feature_cost = args

    best_delta = 0
    best = None

    for ri, rj in room_pairs:
        occ1 = state_snapshot[ri]
        occ2 = state_snapshot[rj]
        if not occ1 or not occ2:
            continue

        actual1 = len(occ1) - 1
        actual2 = len(occ2) - 1

        for s1 in occ1:
            old_ls1 = sum(compat[s1][o] for o in occ1 if o != s1)
            old_rp1 = abs(rpref[s1] - actual1)
            old_fc1 = feature_cost[s1][ri]

            for s2 in occ2:
                old_ls2 = sum(compat[s2][o] for o in occ2 if o != s2)
                old_rp2 = abs(rpref[s2] - actual2)
                old_fc2 = feature_cost[s2][rj]

                old = old_ls1 + old_rp1 + old_fc1 + old_ls2 + old_rp2 + old_fc2

                new_ls2 = sum(compat[s2][o] for o in occ1 if o != s1)
                new_rp2 = abs(rpref[s2] - actual1)
                new_fc2 = feature_cost[s2][ri]

                new_ls1 = sum(compat[s1][o] for o in occ2 if o != s2)
                new_rp1 = abs(rpref[s1] - actual2)
                new_fc1 = feature_cost[s1][rj]

                new = new_ls2 + new_rp2 + new_fc2 + new_ls1 + new_rp1 + new_fc1

                delta = new - old
                if delta < best_delta:
                    best_delta = delta
                    best = (delta, s1, ri, s2, rj)

    return best

def _chunk_room_pairs(n_r, n_chunks):
    all_pairs = [(ri, rj) for ri in range(n_r) for rj in range(ri + 1, n_r)]
    size = max(1, len(all_pairs) // n_chunks)
    return [all_pairs[i:i + size] for i in range(0, len(all_pairs), size)]

def steepest_ascent_hill_climbing(students, rooms, n_workers=None):
    if n_workers is None:
        n_workers = max(1, cpu_count() - 2)
    print(f"Using {n_workers} worker processes (cpu_count={cpu_count()})")

    (s_ids, r_ids, s_idx, r_idx,
     sleep, clean, noise, rpref, cap,
     compat, feature_cost) = build_arrays(students, rooms)

    n_r = len(r_ids)
    state = random_assignment_arrays(len(s_ids), cap)
    print("Calculating initial score...")
    current_value = calculate_value_arrays(state, rpref, compat, feature_cost)
    print(f"Initial score: {current_value}")

    scores = [current_value]
    iteration = 1

    chunks = _chunk_room_pairs(n_r, n_workers)
    print(f"Total room pairs: {sum(len(c) for c in chunks):,}  |  "
          f"Chunks: {len(chunks)}  |  Avg chunk size: {sum(len(c) for c in chunks)//len(chunks):,}")
    print("Starting hill climbing loop...\n")

    interrupted = False
    with Pool(processes=n_workers) as pool:
        try:
            while True:
                print(f"[Iter {iteration}] Score: {current_value} | Dispatching {len(chunks)} chunks to {n_workers} workers...")

                worker_args = [
                    (chunk, [list(occ) for occ in state], rpref, compat, feature_cost)
                    for chunk in chunks
                ]

                results = pool.map(_evaluate_chunk, worker_args)
                print(f"[Iter {iteration}] Workers done. Collecting best swap...")

                global_best = None
                for r in results:
                    if r is not None:
                        if global_best is None or r[0] < global_best[0]:
                            global_best = r

                if global_best is None:
                    print(f"[Iter {iteration}] No improving swap found. Local optimum reached.")
                    break

                delta, s1, r1, s2, r2 = global_best
                state[r1].remove(s1)
                state[r1].append(s2)
                state[r2].remove(s2)
                state[r2].append(s1)
                current_value += delta
                scores.append(current_value)
                print(f"[Iter {iteration}] ✓ Swapped student {s_ids[s1]} (room {r_ids[r1]}) ↔ "
                      f"student {s_ids[s2]} (room {r_ids[r2]}) | Δ={delta:+} | New score: {current_value}\n")
                iteration += 1

        except KeyboardInterrupt:
            print(f"\n⚠  Interrupted at iteration {iteration} (score={current_value}). "
                  "Saving current best…")
            pool.terminate()
            interrupted = True

    print(f"\nHill climbing finished after {iteration} iteration(s). Final score: {current_value}")
    print("Saving score plot...")
    plt.plot(scores)
    plt.xlabel("Iteration")
    plt.ylabel("Score (lower = better)")
    plt.title("Parallel Hill Climbing Score Over Time")
    plt.savefig("score_plot.png")
    plt.show()

    assignment = {}
    for ri, occ in enumerate(state):
        assignment[r_ids[ri]] = [s_ids[si] for si in occ]
    return assignment, interrupted

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
            has = {f[len("has_"):] for f in rooms[room]["features"] if f.startswith("has_")}
            score += len(wanted - has)
    return score

def save_results(output_path, assignment, students, rooms, interrupted=False):
    print(f"Writing results to {output_path}...")
    final_score = calculate_value(assignment, students, rooms)
    status = "INTERRUPTED – partial result" if interrupted else "Completed"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Parallel Hill Climbing Results\n")
        f.write("==============================\n")
        f.write(f"Status: {status}\n")
        f.write(f"Final score: {final_score}\n\n")
        for room_id in sorted(assignment.keys()):
            room = rooms[room_id]
            occupants = assignment[room_id]
            f.write(f"Room {room_id} ({room['name']}) - {len(occupants)}/{room['capacity']} occupants\n")
            for sid in occupants:
                f.write(f"  - {sid}: {students[sid]['name']}\n")
            if not occupants:
                f.write("  - <empty>\n")
            f.write("\n")

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    output_path = root / "output.txt"

    print("Loading input CSV files...")
    students = load_students_csv(data_dir / "students.csv")
    rooms = load_rooms_csv(data_dir / "rooms.csv")
    print(f"Loaded {len(students)} students and {len(rooms)} rooms.")

    assignment, interrupted = steepest_ascent_hill_climbing(students, rooms)
    best_score = calculate_value(assignment, students, rooms)
    print(f"Final score: {best_score}")
    save_results(output_path, assignment, students, rooms, interrupted=interrupted)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()