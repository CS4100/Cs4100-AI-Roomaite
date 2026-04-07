import pandas as pd
from algo import make_student, stochastic_hill_climbing, print_rooms
import os
import matplotlib.pyplot as plt


FEATURE_COLUMNS = {
    "wants_ac": "ac",
    "wants_private_bath": "private_bath",
    "wants_balcony": "balcony",
    "wants_kitchen": "kitchen",
    "wants_laundry": "laundry",
    "wants_wifi": "wifi",
    "wants_parking": "parking",
}


def load_students(path):
    df = pd.read_csv(path)
    students = []

    for _, row in df.iterrows():
        features = [
            feature_name
            for col, feature_name in FEATURE_COLUMNS.items()
            if row[col] == 1
        ]

        students.append(
            make_student(
                student_id=int(row["student_id"]),
                name=str(row["name"]),
                sleep=int(row["sleep"]),
                clean=int(row["clean"]),
                noise=int(row["noise"]),
                roommate=int(row["roommate_count"]),
                features=features,
            )
        )
    return students


def load_room_capacities(path, num_students):
    df = pd.read_csv(path)

    capacities = []
    total = 0

    for cap in df["capacity"]:
        cap = int(cap)
        if total + cap <= num_students:
            capacities.append(cap)
            total += cap
        if total == num_students:
            break

    if total != num_students:
        raise ValueError(f"Got total capacity {total}, need {num_students}")

    return capacities


def main():
    students = load_students("../data/students.csv")
    room_capacities = load_room_capacities("../data/rooms.csv", len(students))

    print("students loaded:", len(students))
    print("room count:", len(room_capacities))
    print("total capacity:", sum(room_capacities))

    rooms, cost, history = stochastic_hill_climbing(
        students=students,
        room_capacities=room_capacities,
        max_steps=50000,
        restarts=1,
        seed=42,
    )



    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(history)), history, linewidth=0.8)
    plt.xlabel("iteration")
    plt.ylabel("total cost")
    plt.title(
        f"stochastic hill climbing - {len(students)} students, "
        f"{len(room_capacities)} rooms"
    )
    plt.tight_layout()
    plt.savefig("results/shc_cost_history.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Best cost found: {cost}\n")
    print_rooms(rooms)


if __name__ == "__main__":
    main()