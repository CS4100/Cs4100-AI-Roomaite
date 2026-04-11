"""
filename: first_choice_hill_climbing.py
First choice hill climbing algorithm for roommate assignment system
"""

import random
import time
from pathlib import Path
from utils import (
    load_students_csv,
    load_rooms_csv,
    build_arrays,
    random_assignment_arrays,
    calculate_value_arrays,
    calculate_value,
    plot_history,
    save_results,
    sample_room_assignments,
)


def first_choice_hill_climbing(students, rooms, show_progress=True, max_iterations=5000):
    """
    Run first-choice hill climbing to find a low-cost room assignment

    At each iteration, randomly shuffles room pairs and student pairs accepting first swap
    that reduces the total objective cost
    """
    # Unpack arrays
    (s_ids, r_ids, s_idx, r_idx,
     sleep, clean, noise, rpref, cap,
     compat, feature_cost) = build_arrays(students, rooms)

    n_r = len(r_ids)

    # Generate a random starting assignment and compute its initial score
    state = random_assignment_arrays(len(s_ids), cap, show_progress=show_progress)
    current_value = calculate_value_arrays(state, rpref, compat, feature_cost)

    # Track the score after every accepted swap for plotting
    history = [current_value]

    for iteration in range(1, max_iterations + 1):
        if show_progress:
            print(f"Iteration {iteration}: score={current_value}")

        # Ensure improved swap
        improved = False

        # Shuffle the room order so it doesn't start on the same one
        room_indices = list(range(n_r))
        random.shuffle(room_indices)

        for ri in room_indices:
            if improved:
                # Stop as soon as swap is accepted
                break

            occ1 = state[ri]
            if not occ1:
                # Skip rooms that are empty
                continue

            # Shuffle second room order
            other_rooms = list(range(n_r))
            random.shuffle(other_rooms)

            for rj in other_rooms:
                if improved:
                    break
                # Cannot swap with itself
                if rj == ri:
                    continue

                # Skip empty rooms
                occ2 = state[rj]
                if not occ2:
                    continue

                # Number of other roommates in each room
                actual1 = len(occ1) - 1
                actual2 = len(occ2) - 1

                # Shuffle students
                candidates1 = list(occ1)
                candidates2 = list(occ2)
                random.shuffle(candidates1)
                random.shuffle(candidates2)

                for s1 in candidates1:
                    if improved:
                        break

                    # Student 1's current contribution to score
                    old_lifestyle_s1 = sum(compat[s1][o] for o in occ1 if o != s1)
                    old_rpref_s1 = abs(rpref[s1] - actual1)
                    old_feat_s1 = feature_cost[s1][ri]

                    for s2 in candidates2:
                        # Student 2's current contribution to score
                        old_lifestyle_s2 = sum(compat[s2][o] for o in occ2 if o != s2)
                        old_rpref_s2 = abs(rpref[s2] - actual2)
                        old_feat_s2 = feature_cost[s2][rj]

                        # Total cost of 1 and 2 before swap
                        old = (old_lifestyle_s1 + old_rpref_s1 + old_feat_s1
                               + old_lifestyle_s2 + old_rpref_s2 + old_feat_s2)

                        # Possible contribution if S2 switched rooms
                        new_lifestyle_s2_in_ri = sum(compat[s2][o] for o in occ1 if o != s1)
                        new_rpref_s2 = abs(rpref[s2] - actual1)
                        new_feat_s2 = feature_cost[s2][ri]

                        # Possible contribution if S1 switched rooms
                        new_lifestyle_s1_in_rj = sum(compat[s1][o] for o in occ2 if o != s2)
                        new_rpref_s1 = abs(rpref[s1] - actual2)
                        new_feat_s1 = feature_cost[s1][rj]

                        # total cost of Student 1 and 2 after switch
                        new = (new_lifestyle_s2_in_ri + new_rpref_s2 + new_feat_s2
                               + new_lifestyle_s1_in_rj + new_rpref_s1 + new_feat_s1)

                        # If delta < 0: swap minimizes cost
                        delta = new - old

                        if delta < 0:
                            # Accept swap
                            state[ri].remove(s1)
                            state[ri].append(s2)
                            state[rj].remove(s2)
                            state[rj].append(s1)
                            current_value += delta
                            history.append(current_value)
                            improved = True
                            if show_progress:
                                print(f"  Accepted swap, new score={current_value}")
                            break

        if not improved:
            # No improvement found, local optimum
            if show_progress:
                print(f"Stopped at iteration {iteration}: local optimum")
            break

    else:
        # Loop completed and no local optima
        if show_progress:
            print(f"Stopped after max_iterations={max_iterations}.")

    # Convert array to original student and room IDs
    assignment = {}
    for ri, occ in enumerate(state):
        assignment[r_ids[ri]] = [s_ids[si] for si in occ]
    return assignment, history


def main():
    # Load data
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    output_path = root / "results" / "output.txt"
    results_dir = root / "results"
    # Make new folder unless exists
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "first_choice_hill_climbing.png"

    # Load files
    print("Loading input CSV files")
    students = load_students_csv(data_dir / "students.csv")
    rooms = load_rooms_csv(data_dir / "rooms.csv")
    # Successful load
    print(f"Loaded {len(students)} students and {len(rooms)} rooms")

    # Run algorithm and time!
    start = time.time()
    assignment, history = first_choice_hill_climbing(
        students, rooms, show_progress=True, max_iterations=5000)
    elapsed = time.time() - start

    # Sample room assignments
    sample_room_assignments(assignment, students, rooms)

    # Report score and save results
    best_score = calculate_value(assignment, students, rooms)
    print(f"Final score: {best_score}")
    save_results(output_path, assignment, students, rooms)
    print(f"Saved results to {output_path}")

    # Plot score over time and save
    plot_history(history, len(students), len(rooms), elapsed, results_path)

if __name__ == "__main__":
    main()