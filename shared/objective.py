"""
objective.py - shared cost function for roommate matching
cs4100 ai project

everyone imports from here so all our algorithms use the same
cost function and we can actually compare results
"""

import numpy as np
import pandas as pd


# columns we compare between students in a room
PREF_COLS = ["sleep", "clean", "noise"]

# feature stuff - what students want vs what rooms have
FEATURE_NAMES = ["ac", "private_bath", "balcony", "kitchen", "laundry", "wifi", "parking"]
STUDENT_FEATURE_COLS = [f"wants_{f}" for f in FEATURE_NAMES]
ROOM_FEATURE_COLS = [f"has_{f}" for f in FEATURE_NAMES]


def load_data(students_path, rooms_path):
    """load both csvs and pull out numpy arrays for fast math"""
    students_df = pd.read_csv(students_path)
    rooms_df = pd.read_csv(rooms_path)

    # just the columns we need for pairwise comparison
    student_prefs = students_df[PREF_COLS].values.astype(int)

    # how many roommates each student wants
    roommate_pref = students_df["roommate_count"].values.astype(int)

    # binary feature columns
    student_features = students_df[STUDENT_FEATURE_COLS].values.astype(int)
    room_features = rooms_df[ROOM_FEATURE_COLS].values.astype(int)
    room_capacities = rooms_df["capacity"].values.astype(int)

    return (students_df, rooms_df, student_prefs, roommate_pref,
            student_features, room_features, room_capacities)


def room_cost(room_idx, student_indices, student_prefs, roommate_pref,
              student_features, room_features):
    """
    cost for one room. three parts:
    1. pairwise compatibility - |sleep| + |clean| + |noise| for every pair
    2. roommate count - penalty if student wants 2 roommates but has 3 etc
    3. feature mismatch - student wants ac but room doesnt have it
    """
    cost = 0
    n = len(student_indices)
    actual_roommates = n - 1

    for i in range(n):
        si = student_indices[i]

        # compare this student with every other student in the room
        for j in range(i + 1, n):
            sj = student_indices[j]
            cost += int(np.sum(np.abs(student_prefs[si] - student_prefs[sj])))

        # how far off is the roommate count from what they wanted
        cost += abs(int(roommate_pref[si]) - actual_roommates)

        # features they want but room doesnt have
        wants = student_features[si]
        has = room_features[room_idx]
        cost += int(np.sum(wants * (1 - has)))

    return cost


def assignment_cost(assignment, student_prefs, roommate_pref,
                    student_features, room_features):
    """total cost across all rooms"""
    total = 0
    for room_idx, student_indices in assignment.items():
        total += room_cost(room_idx, student_indices, student_prefs,
                          roommate_pref, student_features, room_features)
    return total