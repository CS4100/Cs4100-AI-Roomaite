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

For each pair of students, the cost includes:

- difference in sleep preference
- difference in cleanliness preference
- difference in noise preference
- difference in preferred roommate count
- mismatch in room feature preferences

There is also a bonus reduction if two students are preferred roommates.

The overall room assignment cost is:

- sum of pair costs within each room
- summed across all rooms

Lower total cost means a better roommate assignment.


## Files
### `algo.py`
Contains:
- the `Student` data structure
- cost functions
- random initial assignment
- random neighbor generation
- stochastic hill climbing algorithm
- helper functions for printing assignments

### `run_data.py`
Loads data from:
- `data/students.csv`
- `data/rooms.csv`

Then it:
- converts CSV rows into `Student` objects
- runs stochastic hill climbing
- records cost history
- plots cost over iterations
- saves the figure to the `results/` folder

### `data/students.csv`
Student preference dataset.

### `data/rooms.csv`
Room capacity dataset.


## How to Run

From inside the `StochasticHillClimbing` folder:

```bash
python3 run_data.py