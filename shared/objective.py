"""
objective.py - Shared cost function for roommate matching
CS4100 AI Project

everyone imports from here so we're all using the same 
cost function. uses numpy arrays for speed on 10k students
"""

import numpy as np
import pandas as pd

# columns we compare between students
COMPARE_COLS = [
    "sleep", "clean", "noise", "roommate_count",
    "wants_ac", "wants_private_bath", "wants_balcony",
    "wants_kitchen", "wants_laundry", "wants_wifi", "wants_parking"
]


def load_student_matrix(filepath):
    """
    load students csv and return the dataframe + a numpy matrix
    of just the columns we need for cost calculation.
    way faster than using pandas rows in a loop
    """
    df = pd.read_csv(filepath)
    matrix = df[COMPARE_COLS].values.astype(int)
    return df, matrix


def pair_cost(matrix, i, j):
    """cost between two students using precomputed matrix"""
    return int(np.sum(np.abs(matrix[i] - matrix[j])))


def total_cost(matrix, pairs):
    """sum of pair costs across all pairs"""
    total = 0
    for i, j in pairs:
        total += int(np.sum(np.abs(matrix[i] - matrix[j])))
    return total
