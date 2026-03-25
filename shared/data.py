"""
data.py - generate and load student data
CS4100 AI Project

generates random students for testing. no LLM involved,
just random.randint with realistic ranges.
also has csv save/load so the whole team uses the same dataset
"""

import csv
import random
from shared.objective import Student


FEATURES = ["AC", "private_bath", "balcony", "kitchen", "laundry", "wifi", "parking"]


def generate_students(n=20, seed=42):
    """
    make n random students with a fixed seed so 
    everyone on the team gets the exact same data
    """
    rng = random.Random(seed)
    students = []
    for i in range(n):
        s = Student(
            name=f"Student_{i+1}",
            sleep=rng.randint(1, 5),
            clean=rng.randint(1, 5),
            noise=rng.randint(1, 5),
            roommate_count=rng.randint(1, 4),
            room_features=set(rng.sample(FEATURES, rng.randint(1, 4)))
        )
        students.append(s)
    return students


def save_csv(students, filepath):
    """save students to csv so we can share one dataset"""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "sleep", "clean", "noise", "roommate_count", "room_features"])
        for s in students:
            writer.writerow([
                s.name, s.sleep, s.clean, s.noise,
                s.roommate_count, ";".join(sorted(s.room_features))
            ])


def load_csv(filepath):
    """load students from csv"""
    students = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = Student(
                name=row["name"],
                sleep=int(row["sleep"]),
                clean=int(row["clean"]),
                noise=int(row["noise"]),
                roommate_count=int(row["roommate_count"]),
                room_features=set(row["room_features"].split(";")) if row["room_features"] else set()
            )
            students.append(s)
    return students


# generate and save a default dataset if run directly
if __name__ == "__main__":
    students = generate_students(20)
    save_csv(students, "data/students.csv")
    print(f"saved {len(students)} students to data/students.csv")
    for s in students:
        print(f"  {s.name}: sleep={s.sleep} clean={s.clean} noise={s.noise} "
              f"roommates={s.roommate_count} features={s.room_features}")