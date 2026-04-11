# CS4100 AI Roommate Assignment – Stochastic Hill Climbing

## Project Overview
This project solves a roommate assignment problem using **stochastic hill climbing**.

The goal is to assign students to rooms in a way that minimizes total dissatisfaction.  
Each student has several preference attributes, including:

- sleep schedule
- cleanliness
- noise tolerance
- preferred number of roommates
- room feature preferences

The algorithm starts from a random assignment and repeatedly makes small random changes to improve the overall assignment quality.


## Algorithm Used
We use **stochastic hill climbing**.

### Neighbor definition
At each iteration, the algorithm generates **one random neighbor**:
1. randomly choose two different rooms
2. randomly choose one student from each room
3. swap the two students

### Move rule
After generating the neighbor:
- compute the total cost of the new assignment
- **accept the move only if the new cost is lower**

This makes the algorithm a hill climbing method, because it only moves to better states.


## Objective Function
The total cost is based on pairwise incompatibility between students in the same room.

score = pairwise incompatibility + roommate-count mismatch penalty + room-feature mismatch penalty

Lower score means a better roommate assignment.

1. Pairwise incompatibility

- absolute difference in sleep preference
- absolute difference in cleanliness preference
- absolute difference in noise tolerance

2. Roommate-count mismatch penalty

For each student, the algorithm adds: |preferred roommate count - actual roommate count|where: actual roommate count = room size - 1

3. Room-feature mismatch penalty
For each student, the algorithm adds: +1 for each preferred room feature that the assigned room does not have


## Files
### `algo.py`
This file contains the full stochastic hill climbing implementation, including:
- data loading
- objective function
- initial random assignment
- stochastic hill climbing search
- result printing
- plotting

### `generate_data.py`
This file generates the synthetic datasets used in the project:
- data/students.csv
- data/rooms.csv

### `data/students.csv`
Contains student preference data, including:
- student_id
- name
- sleep
- clean
- noise
- roommate_count
- preferred room features such as wants_ac, wants_wifi, etc.

### `data/rooms.csv`
Contains room data, including:
- room_id
- room_name
- capacity
- available room features such as has_ac, has_wifi, etc.

### `results/shc_cost_history.png`
Saved plot showing how the best-so-far cost changes over iterations.

## How to Run
1. If the CSV files already exist:
python3 algo.py

2. If need to regenerate the datasets first:
python3 generate_data.py
python3 algo.py

## Output
### The program prints:
- starting cost
- final cost
- percentage improvement
- number of iterations
- runtime
- number of rooms used
- sample room assignments for the first 10 rooms

### Each sample assignment includes:
- real room name
- room capacity
- assigned student names
- room cost