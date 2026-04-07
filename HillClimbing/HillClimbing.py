import random

def random_assignment(students, rooms):
    assignment = {room: [] for room in rooms}
    student_ids = list(students.keys())
    random.shuffle(student_ids)
    room_list = list(rooms.keys())

    for student in student_ids:
        placed = False

        while not placed:
            room = random.choice(room_list)
            if len(assignment[room]) < rooms[room]["capacity"]:
                assignment[room].append(student)
                placed = True

    return assignment

def room_feature_match(student, room):
    preferred = set(student["room_features"])
    actual = set(room["features"])
    matching = len(preferred.intersection(actual))

    return abs(len(preferred) - matching)

def calculate_value(state):
    score = 0

    for room, occupants in state.items():
        for i in range(len(occupants)):
            for j in range(i+1, len(occupants)):
                s1 = students[occupants[i]]
                s2 = students[occupants[j]]
                score += abs(s1["sleep"] - s2["sleep"])
                score += abs(s1["clean"] - s2["clean"])
                score += abs(s1["noise"] - s2["noise"])

    for room, occupants in state.items():
        room_size = len(occupants)
        for student_id in occupants:
            preferred = students[student_id]["roommate_preference"]
            actual = room_size - 1
            score += abs(preferred - actual)

    for room, occupants in state.items():
        for student_id in occupants:
            student = students[student_id]
            score += room_feature_match(student, rooms[room])

    return score

def get_neighbors(state):
    return 

# current => dict => room: [students] 
def steepest_ascent_hill_climbing(students, rooms):
    current = None # Do i randomly pick a first assignment here?

    while True:
        neighbors = get_neighbors(current)
        best_neighbor = None
        best_value = calculate_value(current)

        for neighbor in neighbors:
            neighbor_value = calculate_value(neighbor)
            if neighbor_value < best_value:
                best_neighbor = neighbor
                best_value = neighbor_value

        if best_neighbor is None or best_value >= calculate_value(current):
            return current

        current = best_neighbor

print(steepest_ascent_hill_climbing(students, rooms))