from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Student:
    """
    Student profile used by the roommate assignment solver.

    Numeric attributes are assumed to be on the same bounded preference scale
    such as 1-5. Higher or lower values do not matter by themselves; only the
    absolute distance between two students matters for compatibility.
    """

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
    """
    Objective function for one pair of students.

    Implemented from the project objective:
        abs(sleep - sleep)
      + abs(clean - clean)
      + abs(noise - noise)
      + abs(roommate# - roommate#)
      + abs(# same features of room - # same features of room)

    The final term in the proposal is ambiguous as written. To make it usable,
    we interpret it as a penalty for *not* sharing room features/preferences.
    If two students share more preferred room features, their penalty is lower.

    Specifically:
        feature_penalty = max_feature_count - shared_feature_count

    so the full pair objective becomes the sum of the four numeric differences
    plus this feature penalty.
    """
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

    # Optional small bonus if a preferred roommate request is satisfied.
    if s2.student_id in s1.preferred_roommates or s1.student_id in s2.preferred_roommates:
        cost -= w["preferred_roommate_bonus"]

    return cost


def room_cost(room: Sequence[Student], weights: Dict[str, int] | None = None) -> int:
    return sum(pair_cost(a, b, weights) for a, b in itertools.combinations(room, 2))


def total_cost(rooms: Sequence[Sequence[Student]], weights: Dict[str, int] | None = None) -> int:
    return sum(room_cost(room, weights) for room in rooms)


def random_assignment(students: Sequence[Student], room_capacities: Sequence[int]) -> List[List[Student]]:
    if sum(room_capacities) != len(students):
        raise ValueError("Total room capacity must equal the number of students.")

    shuffled = list(students)
    random.shuffle(shuffled)

    rooms: List[List[Student]] = []
    start = 0
    for capacity in room_capacities:
        rooms.append(shuffled[start : start + capacity])
        start += capacity
    return rooms


def get_neighbors(rooms: Sequence[Sequence[Student]]) -> List[List[List[Student]]]:
    """
    Neighbor generation by swapping one student from room i with one student from room j.
    This preserves feasibility and room capacities.
    """
    neighbors: List[List[List[Student]]] = []
    current = [list(room) for room in rooms]

    for r1 in range(len(current)):
        for r2 in range(r1 + 1, len(current)):
            for i in range(len(current[r1])):
                for j in range(len(current[r2])):
                    new_rooms = [list(room) for room in current]
                    new_rooms[r1][i], new_rooms[r2][j] = new_rooms[r2][j], new_rooms[r1][i]
                    neighbors.append(new_rooms)

    return neighbors


def stochastic_hill_climbing(
    students: Sequence[Student],
    room_capacities: Sequence[int],
    max_steps: int = 500,
    restarts: int = 25,
    weights: Dict[str, int] | None = None,
    seed: int | None = 42,
) -> Tuple[List[List[Student]], int]:
    """
    Stochastic hill climbing with random restarts.

    At each step, choose randomly among the improving neighbors instead of always
    taking the single best move. Random restarts reduce the chance of getting stuck
    in poor local minima.
    """
    if seed is not None:
        random.seed(seed)

    best_overall_rooms: List[List[Student]] | None = None
    best_overall_cost = float("inf")

    for _ in range(restarts):
        current = random_assignment(students, room_capacities)
        current_cost = total_cost(current, weights)

        for _ in range(max_steps):
            neighbors = get_neighbors(current)
            improving_neighbors: List[Tuple[List[List[Student]], int]] = []

            for neighbor in neighbors:
                cost = total_cost(neighbor, weights)
                if cost < current_cost:
                    improving_neighbors.append((neighbor, cost))

            if not improving_neighbors:
                break

            current, current_cost = random.choice(improving_neighbors)

        if current_cost < best_overall_cost:
            best_overall_rooms = current
            best_overall_cost = current_cost

    if best_overall_rooms is None:
        raise RuntimeError("Solver failed to produce an assignment.")

    return best_overall_rooms, int(best_overall_cost)


def explain_assignment(rooms: Sequence[Sequence[Student]], weights: Dict[str, int] | None = None) -> str:
    lines: List[str] = []
    grand_total = total_cost(rooms, weights)
    lines.append(f"Total objective cost: {grand_total}")

    for idx, room in enumerate(rooms, start=1):
        names = ", ".join(student.name for student in room)
        rcost = room_cost(room, weights)
        lines.append(f"Room {idx} ({len(room)} students) -> {names} | room cost = {rcost}")

        for a, b in itertools.combinations(room, 2):
            shared = sorted(a.features & b.features)
            lines.append(
                f"  Pair {a.name} / {b.name}: cost={pair_cost(a, b, weights)}, "
                f"shared_features={shared if shared else 'none'}"
            )

    return "\n".join(lines)


def print_rooms(rooms: Sequence[Sequence[Student]], weights: Dict[str, int] | None = None) -> None:
    print(explain_assignment(rooms, weights))


def demo_students() -> List[Student]:
    """Example dataset for presentation/demo purposes."""
    return [
        make_student(1, "Alice", 1, 5, 1, 1, ["quiet", "study", "AC"], [2]),
        make_student(2, "Ben", 1, 4, 1, 1, ["quiet", "study", "AC"], [1]),
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

    rooms, cost = stochastic_hill_climbing(
        students=students,
        room_capacities=room_capacities,
        max_steps=300,
        restarts=30,
        seed=42,
    )

    print(f"Best cost found: {cost}\n")
    print_rooms(rooms)