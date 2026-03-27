# Dorm Room Assignment with Stochastic Hill Climbing

## Project idea
This project assigns students to dorm rooms by minimizing roommate mismatch using **stochastic hill climbing**, a local search algorithm. The goal is to reduce dissatisfaction by grouping students with similar living preferences.

The system is designed for a university housing setting where students are assigned to a fixed layout of rooms with predefined capacities.

## Objective function
For every pair of students in the same room, we compute:

\[
|sleep_i - sleep_j| + |clean_i - clean_j| + |noise_i - noise_j| + |roommate_i - roommate_j| + feature\_penalty
\]

The last term in the proposal was written as:

\[
|\#\text{same room features} - \#\text{same room features}|
\]

Since that expression is ambiguous, the implementation interprets it as a compatibility penalty based on **shared room-feature preferences**:

\[
feature\_penalty = \max(|F_i|, |F_j|, 1) - |F_i \cap F_j|
\]

This means that students who share more room preferences, such as wanting a quiet room, AC, a window, or late-night social space, receive a lower cost.

The total objective value is the sum of all pairwise costs across all rooms. Lower cost means a better housing assignment.

## Why stochastic hill climbing?
Regular hill climbing always picks the single best improvement, which can get trapped quickly in a local minimum. **Stochastic hill climbing** instead chooses randomly among improving moves. This adds exploration while still moving toward lower cost assignments.

To further improve results, this implementation uses **random restarts**:
1. Generate a random valid assignment
2. Repeatedly swap students across rooms
3. Randomly choose one improving swap
4. Stop when no improving neighbor exists
5. Restart from a new random assignment several times
6. Return the best overall result

## State representation
- A **state** is a complete room assignment.
- Each room has a fixed capacity.
- Every student appears exactly once.

Example:
```python
[
    [Alice, Ben],
    [Cara, David],
    [Eva, Finn],
    [Gina, Hugo]
]