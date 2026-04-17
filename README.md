# CS4100 AI Roommate Assignment – Search Algorithms for Student Housing

## Project Overview
This project solves a roommate assignment problem using **four search algorithms**:

- simulated annealing
- steepest-ascent hill climbing
- first choice hill climbing
- stochastic hill climbing

The goal is to assign students to rooms in a way that minimizes total dissatisfaction.

Each student has several preference attributes, including:

- sleep schedule
- cleanliness
- noise tolerance
- preferred number of roommates
- room feature preferences

Each room also has its own capacity and available features.

All four algorithms use the **same dataset** and the **same objective function**, so their performance can be compared fairly.


## Algorithms Used

### 1. Simulated Annealing
Simulated annealing starts from a random assignment and repeatedly swaps students between rooms.

#### Neighbor definition
At each iteration:
1. randomly choose two different rooms
2. randomly choose one student from each room
3. swap the two students

#### Move rule
After generating the neighbor:
- if the new cost is lower, always accept it
- if the new cost is higher, sometimes still accept it with probability:

e^(-Δ / T)

where:
- Δ is the increase in cost
- T is the current temperature

#### Cooling rule
The temperature gradually decreases over time by multiplying it by a cooling factor.

This helps the algorithm:
- explore more at the beginning
- become more selective later

#### Key idea
Simulated annealing can escape local optima by accepting worse moves early.


### 2. Steepest-Ascent Hill Climbing
Steepest-ascent hill climbing also starts from a random assignment, but instead of making one random move, it checks many possible neighbors and chooses the **best** one.

#### Neighbor definition
The algorithm considers swaps between students in different rooms.

#### Move rule
At each step:
- evaluate all candidate swaps
- find the one that gives the largest improvement
- make that single best move
- stop when no improving move exists

#### Key idea
This method is more thorough than the other hill climbing methods, but it is also much more expensive because evaluating all neighbors takes a very long time.

#### Note
In this project, steepest-ascent hill climbing had the highest computational cost and was much slower than the other approaches.


### 3. First Choice Hill Climbing
First choice hill climbing starts from a random assignment and tests random neighbors one at a time.

#### Neighbor definition
At each iteration:
1. randomly choose two different rooms
2. randomly choose one student from each room
3. swap the two students

#### Move rule
After generating a neighbor:
- if the swap improves the score, accept it immediately
- otherwise reject it and keep searching

This means the algorithm does **not** search for the best possible move.  
It just takes the **first improving move** it finds.

#### Key idea
First choice hill climbing is faster per step than steepest-ascent hill climbing because it avoids checking every possible neighbor.


### 4. Stochastic Hill Climbing
Stochastic hill climbing also starts from a random assignment and repeatedly makes random swaps.

#### Neighbor definition
At each iteration:
1. randomly choose two different rooms
2. randomly choose one student from each room
3. swap the two students

#### Move rule
After generating the neighbor:
- compute the total cost of the new assignment
- accept the move only if the new cost is lower

This makes the algorithm a hill climbing method, because it only moves to better states.

#### Key idea
Stochastic hill climbing is simple and fast, but like other hill climbing methods, it can get stuck in a local optimum.


## Objective Function
The total cost is based on student incompatibility and room mismatch penalties.

score = pairwise incompatibility + roommate-count mismatch penalty + room-feature mismatch penalty

Lower score means a better roommate assignment.

### 1. Pairwise incompatibility
For students in the same room, the algorithm adds:

- absolute difference in sleep preference
- absolute difference in cleanliness preference
- absolute difference in noise tolerance

So students with more similar habits produce a lower cost.

### 2. Roommate-count mismatch penalty
For each student, the algorithm adds:

|preferred roommate count - actual roommate count|

where:

actual roommate count = room size - 1

This penalizes assignments where the number of roommates does not match the student's preference.

### 3. Room-feature mismatch penalty
For each student, the algorithm adds:

+1 for each preferred room feature that the assigned room does not have

Examples of room features include:
- AC
- wifi
- laundry
- and other binary room attributes


## Data

### `students.csv`
This file contains student preference data.

It includes:
- student_id
- name
- sleep
- clean
- noise
- roommate_count
- preferred room features such as wants_ac, wants_wifi, etc.

The project dataset contains **10,000 students**.

### `rooms.csv`
This file contains room information.

It includes:
- room_id
- room_name
- capacity
- available room features such as has_ac, has_wifi, etc.

The project dataset contains **5,000 rooms**.

### Data generation
The dataset is randomly generated using NumPy with a fixed seed so results are reproducible.


## Files
The exact filenames may vary depending on each team member’s implementation, but the project includes code for all four algorithms and the dataset files.

Typical project files include:

### Algorithm files
- simulated annealing implementation
- steepest-ascent hill climbing implementation
- first choice hill climbing implementation
- stochastic hill climbing implementation

Each algorithm file usually contains:
- data loading
- objective function
- initial random assignment
- search logic
- result printing
- plotting or output visualization

### `generate_data.py`
This file generates the synthetic datasets used in the project:
- `data/students.csv`
- `data/rooms.csv`

### `data/students.csv`
Student preference dataset.

### `data/rooms.csv`
Room information dataset.

### Output / result files
Depending on the implementation, the project may also save:
- cost history plots
- comparison graphs
- printed room assignment samples
- runtime summaries

### Presentation file
- `AI_Roommate_Matching_Presentation.pdf`

This presentation summarizes the problem, methods, results, and comparison across algorithms. :contentReference[oaicite:1]{index=1}


## How to Run

### 1. If the CSV files already exist
Run the algorithm file you want to test. For example:

python3 simulated_annealing.py  
python3 hill_climbing.py  
python3 first_choice_hill_climbing.py  
python3 stochastic_hill_climbing.py

### 2. If you need to regenerate the datasets first
Run:

python3 generate_data.py

Then run any of the algorithm files.

> Note: replace the filenames above with your team’s actual filenames if they are different.


## Output
Each algorithm prints a summary of its performance.

### The program may print:
- starting cost
- final cost
- percentage improvement
- number of iterations
- runtime
- number of rooms used
- sample room assignments

### Each sample assignment may include:
- room name
- room capacity
- assigned student names
- room cost


## Results Summary

### Simulated Annealing
- Starting cost: 70,710
- Final cost: 55,483
- Improvement: 21.5%
- Runtime: 5.2s

### First Choice Hill Climbing
- Starting cost: 64,785
- Final cost: 50,842
- Improvement: 21.52%
- Runtime: 111.0s

### Stochastic Hill Climbing
- Starting cost: 70,710
- Final cost: 47,767
- Improvement: 32.4%
- Runtime: 7.41s

### Steepest-Ascent Hill Climbing
Steepest-ascent hill climbing was much slower than the other methods because it tries to evaluate a very large number of possible swaps. In the project presentation, its runtime is described as too large to compare directly on the final comparison slide. :contentReference[oaicite:2]{index=2}


## Comparison
This project compares the four algorithms under the same conditions.

### Main takeaway
Among the tested methods:
- **stochastic hill climbing** gave the best overall result in the final comparison
- **simulated annealing** was also fast and effective
- **first choice hill climbing** improved the assignment, but took longer
- **steepest-ascent hill climbing** was too computationally expensive for this problem size

This shows that for large-scale roommate assignment problems, simpler local search methods with efficient neighbor generation can be more practical than exhaustive search over neighbors.


## Conclusion
This project shows that search algorithms can be used to solve a large roommate assignment problem efficiently.

By modeling student preferences and room features as a cost function, we can compare different optimization strategies on the same task.

The results suggest that:
- local search is a useful approach for large assignment problems
- runtime matters just as much as solution quality
- stochastic methods can perform very well on large real-world style datasets