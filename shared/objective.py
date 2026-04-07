"""
objective.py - Shared cost function for roommate matching
CS4100 AI Project

cost function considers:
1. how compatible the students in a room are with each other
2. how well the room's features match what each student wants
"""

import numpy as np
import pandas as pd
from itertools import combinations


# student preference columns (for comparing students to each other)
PREF_COLS = ["sleep", "clean", "noise", "roommate_count"]

# feature columns - student wants vs room has
FEATURE_NAMES = ["ac", "private_bath", "balcony", "kitchen", "laundry", "wifi", "parking"]
STUDENT_FEATURE_COLS = [f"wants_{f}" for f in FEATURE_NAMES]
ROOM_FEATURE_COLS = [f"has_{f}" for f in FEATURE_NAMES]


def load_data(students_path, rooms_path):
    """
    load both csvs and pull out the numpy arrays we need
    for fast cost calculation
    """
    students_df = pd.read_csv(students_path)
    rooms_df = pd.read_csv(rooms_path)

    # numpy arrays for fast math
    student_prefs = students_df[PREF_COLS].values.astype(int)
    student_features = students_df[STUDENT_FEATURE_COLS].values.astype(int)
    room_features = rooms_df[ROOM_FEATURE_COLS].values.astype(int)
    room_capacities = rooms_df["capacity"].values.astype(int)

    return students_df, rooms_df, student_prefs, student_features, room_features, room_capacities


def student_pair_cost(student_prefs, student_features, i, j):
    """
    compatibility cost between two students
    = sum of absolute differences in preferences
    + number of features they disagree on
    """
    pref_diff = int(np.sum(np.abs(student_prefs[i] - student_prefs[j])))
    feat_diff = int(np.sum(np.abs(student_features[i] - student_features[j])))
    return pref_diff + feat_diff


def room_match_cost(student_features, room_features, student_idx, room_idx):
    """
    how well does a student fit in this room?
    penalty for each feature the student wants but room doesnt have
    """
    wants = student_features[student_idx]
    has = room_features[room_idx]
    # only penalize when student wants it (1) but room doesn't have it (0)
    penalty = int(np.sum(wants * (1 - has)))
    return penalty


def assignment_cost(assignment, student_prefs, student_features, room_features):
    """
    total cost of an assignment
    assignment = dict of {room_idx: [list of student indices]}
    
    cost = sum of all pairwise student costs in each room
         + sum of room feature mismatch penalties for each student
    """
    total = 0

    for room_idx, student_indices in assignment.items():
        # pairwise student compatibility
        for i in range(len(student_indices)):
            for j in range(i + 1, len(student_indices)):
                total += student_pair_cost(
                    student_prefs, student_features,
                    student_indices[i], student_indices[j]
                )

            # room feature mismatch for each student
            total += room_match_cost(
                student_features, room_features,
                student_indices[i], room_idx
            )

    return total