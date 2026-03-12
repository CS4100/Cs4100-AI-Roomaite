import random
from itertools import combinations

# Sample Data
students = [
    {"id": 1, "sleep": 2, "clean": 4, "noise": 1, "roommate_preference": 3, "room_features": ["quiet_floor", "AC"]},
    {"id": 2, "sleep": 3, "clean": 4, "noise": 2, "roommate_preference": 1, "room_features": ["AC"]}
]

rooms = [
    {"room_id": "A101", "capacity": 2, "building": "example", "features": ["AC", "quiet_floor"]},
    {"room_id": "A102", "capacity": 3, "building": "example", "features": ["AC"]}
]

student_map = {s["id"]: s for s in students}
room_map    = {r["room_id"]: r for r in rooms}

# Objective Function
def cost_per_feature(s1, s2, room):
    """
    The dissatisfaction score for two roommates in a given room
    Lower score = better
    """
    room_feats = set(room["features"])

    # The number of features that match student preferred features
    matched1 = len(set(s1["room_features"]) & room_feats)
    matched2 = len(set(s2["room_features"]) & room_feats)

    # Return the difference in each
    return (
        abs(s1["sleep"] - s2["sleep"]) + abs(s1["clean"] - s2["clean"]) + abs(s1["noise"] - s2["noise"]) +
        abs(s1["roommate_preference"] - s2["roommate_preference"]) + abs(matched1 - matched2)
    )

def objective(assignment):
    """
    Total dissatisfaction score across all rooms.
    """
    # Initialize total score as zero
    total = 0
    # Iterate through room and student ids
    for room_id, student_ids in assignment.items():
        room = room_map[room_id]
        occupants = [student_map[sid] for sid in student_ids]
        # Add the total cost of dissatisfaction for students and rooms
        for s1, s2 in combinations(occupants, 2):
            total += cost_per_feature(s1, s2, room)
    return total

# Hill Climbing
def initial_assignment(students, rooms):
    """
    Randomly assigns students to rooms
    """
    # Shuffle students to make a random assignment
    shuffle = random.sample([s["id"] for s in students], len(students))
    # Create empty dictionary to keep assignment with room id
    assignment = {r["room_id"]: [] for r in rooms}
    # Start at 0
    idx = 0
    # For each room in the list
    for room in rooms:
        # Number of students to place in each room
        # Capacity or students left (whichever is smaller)
        num_students = min(room["capacity"], len(shuffle) - idx)
        # Slices list add the students to rooms
        assignment[room["room_id"]] = shuffle[idx : idx + num_students]
        # Index is whatever the number of students place and breaks when complete
        idx += num_students
        if idx >= len(shuffle):
            break
    return assignment

def get_neighbors(assignment):
    """
    Get all neighbors by swapping students
    """
    # Working on this still
    return

def hill_climbing(students, rooms, max_iterations=1000):
    current = initial_assignment(students, rooms)
    current_cost = objective(current)

    for _ in range(max_iterations):
        neighbors = get_neighbors(current)
        if not neighbors:
            break

        # First Choice:
        # Shuffle neighbors and take the first improvement
        random.shuffle(neighbors)
        # There is no improvement yet, so false
        improvement = False
        # Iterate through neighbors
        for neighbor in neighbors:
            # Cost of neighbor
            neighbor_cost = objective(neighbor)
            # Determine if neighbor cost is less than the current
            if neighbor_cost < current_cost:
                # If so current is the neighbor
                current = neighbor
                current_cost = neighbor_cost
                improvement = True
                break

        # If no improvement then there could be a local min
        if not improvement:
            break  # no improvement found
    # Return current and current cost
    return current, current_cost

# Run
if __name__ == "__main__":
    assignment, cost = hill_climbing(students, rooms, max_iterations=1000)
    print(f"Total Dissatisfaction Score: {cost}")
    for room_id, student_ids in assignment.items():
        print(f"  Room {room_id}: Students {student_ids}")