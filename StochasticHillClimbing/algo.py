import random
import copy

def pair_cost(s1, s2):
    return (
        abs(s1["sleep"] - s2["sleep"]) +
        abs(s1["clean"] - s2["clean"]) +
        abs(s1["noise"] - s2["noise"]) +
        abs(s1["roommate"] - s2["roommate"]) +
        abs(s1["feature"] - s2["feature"])
    )

def total_cost(rooms):
    cost = 0

    for room in rooms:
        for i in range(len(room)):
            for j in range(i + 1, len(room)):
                cost += pair_cost(room[i], room[j])

    return cost

def random_assignment(students, room_size):
    shuffled = students[:]
    random.shuffle(shuffled)

    rooms = []
    for i in range(0, len(shuffled), room_size):
        rooms.append(shuffled[i:i + room_size])

    return rooms

def get_neighbors(rooms):
    neighbors = []

    for r1 in range(len(rooms)):
        for r2 in range(r1 + 1, len(rooms)):
            for i in range(len(rooms[r1])):
                for j in range(len(rooms[r2])):
                    new_rooms = copy.deepcopy(rooms)

                    new_rooms[r1][i], new_rooms[r2][j] = new_rooms[r2][j], new_rooms[r1][i]

                    neighbors.append(new_rooms)

    return neighbors

def stochastic_hill_climbing(students, room_size=2, max_steps=200):
    current = random_assignment(students, room_size)
    current_cost = total_cost(current)

    for step in range(max_steps):
        neighbors = get_neighbors(current)

        better_neighbors = []
        for neighbor in neighbors:
            cost = total_cost(neighbor)
            if cost < current_cost:
                better_neighbors.append((neighbor, cost))

        if len(better_neighbors) == 0:
            break

        current, current_cost = random.choice(better_neighbors)

    return current, current_cost

def print_rooms(rooms):
    for i, room in enumerate(rooms):
        names = [student["name"] for student in room]
        print(f"Room {i+1}: {names}")