from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Student:
    student_id: int
    name: str
    sleep: int
    clean: int
    noise: int
    roommate: int
    features: frozenset[str]
    preferred_roommates: frozenset[int] = frozenset()


def make_student(
    student_id: int,
    name: str,
    sleep: int,
    clean: int,
    noise: int,
    roommate: int,
    features: Sequence[str],
    preferred_roommates: Sequence[int] | None = None,
) -> Student:
    return Student(
        student_id=student_id,
        name=name,
        sleep=sleep,
        clean=clean,
        noise=noise,
        roommate=roommate,
        features=frozenset(features),
        preferred_roommates=frozenset(preferred_roommates or []),
    )


DEFAULT_WEIGHTS: Dict[str, int] = {
    "sleep": 1,
    "clean": 1,
    "noise": 1,
    "roommate": 1,
    "feature": 1,
    "preferred_roommate_bonus": 2,
}


def pair_cost(s1: Student, s2: Student, weights: Dict[str, int] | None = None) -> int:
    w = DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)

    shared_features = len(s1.features & s2.features)
    max_feature_count = max(len(s1.features), len(s2.features), 1)
    feature_penalty = max_feature_count - shared_features

    cost = (
        w["sleep"] * abs(s1.sleep - s2.sleep)
        + w["clean"] * abs(s1.clean - s2.clean)
        + w["noise"] * abs(s1.noise - s2.noise)
        + w["roommate"] * abs(s1.roommate - s2.roommate)
        + w["feature"] * feature_penalty
    )

    if s2.student_id in s1.preferred_roommates or s1.student_id in s2.preferred_roommates:
        cost -= w["preferred_roommate_bonus"]

    return cost


def room_cost(room: Sequence[Student], weights: Dict[str, int] | None = None) -> int:
    total = 0
    for a, b in itertools.combinations(room, 2):
        total += pair_cost(a, b, weights)
    return total


def total_cost(rooms: Sequence[Sequence[Student]], weights: Dict[str, int] | None = None) -> int:
    return sum(room_cost(room, weights) for room in rooms)


def random_assignment(students: Sequence[Student], room_capacities: Sequence[int]) -> List[List[Student]]:
    if sum(room_capacities) != len(students):
        raise ValueError("Total room capacity must equal number of students.")

    shuffled = list(students)
    random.shuffle(shuffled)

    rooms: List[List[Student]] = []
    start = 0
    for capacity in room_capacities:
        rooms.append(shuffled[start:start + capacity])
        start += capacity

    return rooms


def random_neighbor(rooms: Sequence[Sequence[Student]]) -> List[List[Student]]:
    """
    Generate exactly ONE random neighbor:
    choose two random rooms, then swap one random student from each.
    """
    new_rooms = [list(room) for room in rooms]

    if len(new_rooms) < 2:
        return new_rooms

    r1, r2 = random.sample(range(len(new_rooms)), 2)

    if not new_rooms[r1] or not new_rooms[r2]:
        return new_rooms

    i = random.randrange(len(new_rooms[r1]))
    j = random.randrange(len(new_rooms[r2]))

    new_rooms[r1][i], new_rooms[r2][j] = new_rooms[r2][j], new_rooms[r1][i]
    return new_rooms


def stochastic_hill_climbing(
    students: Sequence[Student],
    room_capacities: Sequence[int],
    max_steps: int = 50000,
    restarts: int = 1,
    weights: Dict[str, int] | None = None,
    seed: int | None = 42,
) -> Tuple[List[List[Student]], int, List[int]]:
    """
    Stochastic hill climbing:
    - start with a random assignment
    - each iteration generates ONE random neighbor
    - move only if the neighbor is better
    - keep best-so-far history for plotting
    """
    if seed is not None:
        random.seed(seed)

    best_overall_rooms: List[List[Student]] | None = None
    best_overall_cost = float("inf")
    history: List[int] = []

    for _ in range(restarts):
        current_rooms = random_assignment(students, room_capacities)
        current_cost = total_cost(current_rooms, weights)

        if current_cost < best_overall_cost:
            best_overall_rooms = [list(room) for room in current_rooms]
            best_overall_cost = current_cost

        history.append(int(best_overall_cost))

        for _ in range(max_steps):
            neighbor_rooms = random_neighbor(current_rooms)
            neighbor_cost = total_cost(neighbor_rooms, weights)

            if neighbor_cost < current_cost:
                current_rooms = neighbor_rooms
                current_cost = neighbor_cost

                if current_cost < best_overall_cost:
                    best_overall_rooms = [list(room) for room in current_rooms]
                    best_overall_cost = current_cost

            history.append(int(best_overall_cost))

    if best_overall_rooms is None:
        raise RuntimeError("No assignment produced.")

    return best_overall_rooms, int(best_overall_cost), history


def explain_assignment(rooms: Sequence[Sequence[Student]], weights: Dict[str, int] | None = None) -> str:
    lines: List[str] = []
    lines.append(f"Total objective cost: {total_cost(rooms, weights)}")

    for idx, room in enumerate(rooms, start=1):
        names = ", ".join(student.name for student in room)
        lines.append(f"Room {idx} ({len(room)} students): {names} | room cost = {room_cost(room, weights)}")

    return "\n".join(lines)


def print_rooms(rooms: Sequence[Sequence[Student]], weights: Dict[str, int] | None = None) -> None:
    print(explain_assignment(rooms, weights))


def demo_students() -> List[Student]:
    return [
        make_student(1, "Alice", 1, 5, 1, 1, ["quiet", "study", "ac"], [2]),
        make_student(2, "Ben", 1, 4, 1, 1, ["quiet", "study", "ac"], [1]),
        make_student(3, "Cara", 5, 2, 5, 2, ["social", "late-night", "warm"], [4]),
        make_student(4, "David", 4, 2, 4, 2, ["social", "late-night", "warm"], [3]),
        make_student(5, "Eva", 2, 5, 2, 1, ["quiet", "clean", "window"]),
        make_student(6, "Finn", 2, 4, 2, 1, ["quiet", "clean", "window"]),
        make_student(7, "Gina", 4, 1, 5, 3, ["social", "music", "warm"]),
        make_student(8, "Hugo", 4, 2, 4, 3, ["social", "music", "warm"]),
    ]


if __name__ == "__main__":
    students = demo_students()
    room_capacities = [2, 2, 2, 2]

    rooms, cost, history = stochastic_hill_climbing(
        students=students,
        room_capacities=room_capacities,
        max_steps=1000,
        restarts=3,
        seed=42,
    )

    print(f"Best cost found: {cost}\n")
    print_rooms(rooms)
    print(f"\nHistory points: {len(history)}")