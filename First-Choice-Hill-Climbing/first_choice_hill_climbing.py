import random
import pandas as pd
from itertools import combinations

amenities = ["ac", "private_bath", "balcony", "kitchen", "laundry", "wifi", "parking"]

# Load data in from CSVs
def load_students(path="data/students.csv"):
    df = pd.read_csv(path)
    students = []
    for _, row in df.iterrows():
        students.append({
            "id": int(row["student_id"]),
            "name": row["name"],
            "sleep": int(row["sleep"]),
            "clean": int(row["clean"]),
            "noise": int(row["noise"]),
            "roommate_count": int(row["roommate_count"]),
            **{f"wants_{a}": int(row[f"wants_{a}"]) for a in amenities}
        })
    return students

def load_rooms(path="data/rooms.csv"):
    df = pd.read_csv(path)
    rooms = []
    for _, row in df.iterrows():
        rooms.append({
            "room_id": int(row["room_id"]),
            "name": row["room_name"],
            "capacity": int(row["capacity"]),
            **{f"has_{a}": int(row[f"has_{a}"]) for a in amenities}
        })
    return rooms

# Objective functions

def student_room_cost(student, room):
    """Penalty for each amenity the student wants but the room doesn't have."""
    return sum(
        student[f"wants_{a}"] and not room[f"has_{a}"]
        for a in amenities
    )

def student_pair_cost(s1, s2):
    """Lifestyle incompatibility between two roommates."""
    return (
        abs(s1["sleep"] - s2["sleep"]) +
        abs(s1["clean"] - s2["clean"]) +
        abs(s1["noise"] - s2["noise"]) +
        abs(s1["roommate_count"] - s2["roommate_count"])
    )

def objective(assignment, room_map, student_map):
    total = 0
    for room_id, student_ids in assignment.items():
        room = room_map[room_id]
        occupants = [student_map[sid] for sid in student_ids]
        # Pairwise lifestyle compatibility
        for s1, s2 in combinations(occupants, 2):
            total += student_pair_cost(s1, s2)
        # Each student's unmet room preferences
        for s in occupants:
            total += student_room_cost(s, room)
    return total

# First choice Hill Climbing

def initial_assignment(students, rooms):
    shuffle = random.sample([s["id"] for s in students], len(students))
    assignment = {r["room_id"]: [] for r in rooms}
    idx = 0
    for room in rooms:
        n = min(room["capacity"], len(shuffle) - idx)
        assignment[room["room_id"]] = shuffle[idx: idx + n]
        idx += n
        if idx >= len(shuffle):
            break
    return assignment

def get_neighbors(assignment):
    """Generate all neighbors by swapping one student between every pair of rooms."""
    neighbors = []
    room_ids = list(assignment.keys())
    for r1, r2 in combinations(room_ids, 2):
        for i, s1 in enumerate(assignment[r1]):
            for j, s2 in enumerate(assignment[r2]):
                neighbor = {rid: list(sids) for rid, sids in assignment.items()}
                neighbor[r1][i] = s2
                neighbor[r2][j] = s1
                neighbors.append(neighbor)
    return neighbors

def first_choice_hill_climbing(students, rooms, student_map, room_map, max_iterations=1000):
    current = initial_assignment(students, rooms)
    current_cost = objective(current, room_map, student_map)

    for _ in range(max_iterations):
        neighbors = get_neighbors(current)
        if not neighbors:
            break

        random.shuffle(neighbors)
        improvement = False
        for neighbor in neighbors:
            neighbor_cost = objective(neighbor, room_map, student_map)
            if neighbor_cost < current_cost:
                current = neighbor
                current_cost = neighbor_cost
                improvement = True
                break

        if not improvement:
            break

    return current, current_cost

# Run
if __name__ == "__main__":
    students = load_students("data/students.csv")
    rooms = load_rooms("data/rooms.csv")
    student_map = {s["id"]: s for s in students}
    room_map = {r["room_id"]: r for r in rooms}

    assignment, cost = first_choice_hill_climbing(students, rooms, student_map, room_map)

    print(f"Total Dissatisfaction Score: {cost}\n")
    for room_id, student_ids in assignment.items():
        names = [student_map[sid]["name"] for sid in student_ids]
        print(f"{room_map[room_id]['name']}: {', '.join(names)}")