"""
generate_data.py - create student and room datasets
CS4100 AI Project

generates 10k students and rooms as pandas DataFrames
saves to CSV so the whole team can use them
"""

import pandas as pd
import numpy as np

def generate_students(n=10000, seed=42):
    """
    generate n students with random preferences
    all values are random ints
    """
    np.random.seed(seed)
    
    df = pd.DataFrame({
        "student_id": range(1, n + 1),
        "name": [f"Student_{i}" for i in range(1, n + 1)],
        "sleep": np.random.randint(1, 6, size=n),        # 1=early bird, 5=night owl
        "clean": np.random.randint(1, 6, size=n),        # 1=messy, 5=spotless
        "noise": np.random.randint(1, 6, size=n),        # 1=quiet, 5=loud is fine
        "roommate_count": np.random.randint(1, 5, size=n), # how many roommates they want
        "wants_ac": np.random.randint(0, 2, size=n),      # 0=no, 1=yes
        "wants_private_bath": np.random.randint(0, 2, size=n),
        "wants_balcony": np.random.randint(0, 2, size=n),
        "wants_kitchen": np.random.randint(0, 2, size=n),
        "wants_laundry": np.random.randint(0, 2, size=n),
        "wants_wifi": np.random.randint(0, 2, size=n),
        "wants_parking": np.random.randint(0, 2, size=n),
    })
    
    return df


def generate_rooms(n=5000, seed=42):
    """
    generate n rooms with random features
    binary columns: 0 = doesn't have it  1 = has it
    """
    np.random.seed(seed + 1)  # different seed so rooms aren't identical to students
    
    df = pd.DataFrame({
        "room_id": range(1, n + 1),
        "room_name": [f"Room_{i}" for i in range(1, n + 1)],
        "capacity": np.random.choice([1, 2, 3, 4], size=n, p=[0.1, 0.5, 0.3, 0.1]),  # most rooms are doubles/triples
        "has_ac": np.random.randint(0, 2, size=n),
        "has_private_bath": np.random.randint(0, 2, size=n),
        "has_balcony": np.random.randint(0, 2, size=n),
        "has_kitchen": np.random.randint(0, 2, size=n),
        "has_laundry": np.random.randint(0, 2, size=n),
        "has_wifi": np.random.randint(0, 2, size=n),
        "has_parking": np.random.randint(0, 2, size=n),
    })
    
    return df


if __name__ == "__main__":
    # generate and save
    students_df = generate_students(10000)
    rooms_df = generate_rooms(5000)
    
    students_df.to_csv("data/students.csv", index=False)
    rooms_df.to_csv("data/rooms.csv", index=False)
    
    print("students dataset:")
    print(f"  rows: {len(students_df)}")
    print(f"  columns: {list(students_df.columns)}")
    print(students_df.head())
    print()
    print("rooms dataset:")
    print(f"  rows: {len(rooms_df)}")
    print(f"  columns: {list(rooms_df.columns)}")
    print(rooms_df.head())
