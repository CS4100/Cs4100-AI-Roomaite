# DormMatch AI — CS4100

Local search algorithms for optimizing student dorm room assignments.

---

## Problem

Given students with lifestyle preferences and rooms with amenities, find an assignment that minimizes total dissatisfaction across:
- **Lifestyle compatibility** between roommates (sleep, cleanliness, noise)
- **Roommate count preference** vs. actual occupancy
- **Unmet amenity preferences** per student

Lower score = better assignment.

---

## Project Structure

```
Cs4100-AI-Roomaite/
├── data/
│   ├── students.csv
│   └── rooms.csv
├── First-Choice-Hill-Climbing/
│   ├── first_choice_hill_climbing.py
└── output.txt
```

---

## How to Run

```bash
pip install tqdm

python algorithms/first_choice_hill_climbing.py
```

Results are saved to `output.txt`.

---

## Algorithms

| Algorithm | Strategy |
|---|---|
| First Choice Hill Climbing | Accepts the first improving swap found (random order) |

---

## Team

| Name | Algorithm |
|---|---|
| Isabella Uhniat | First Choice Hill Climbing |
