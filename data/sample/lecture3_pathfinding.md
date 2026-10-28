# Lecture 3: Pathfinding and Search Algorithms

## Overview
This lecture covers how AI agents find paths through an environment.
We compare blind search with informed (heuristic) search.

## Breadth-First Search (BFS)
BFS explores the search space level by level, like ripples in a pond.
It visits all states one step away, then two steps away, and so on.
Because it expands in rings, BFS is guaranteed to find the shortest
path when all step costs are equal. The cost: it explores many states
that are nowhere near the goal.

## Depth-First Search (DFS)
DFS commits to one path and follows it as deep as possible, backtracking
only when it gets stuck. It uses less memory than BFS, but the path it
returns can be far from optimal.

## A* Search
A* improves on blind search by combining two values for each state:

    f(n) = g(n) + h(n)

- g(n) is the exact cost already paid to reach state n
- h(n) is a heuristic: an estimate of the remaining distance to the goal

On a grid, a common heuristic is the Manhattan distance — the horizontal
plus vertical distance to the goal, like walking city blocks.

A heuristic is admissible if it never overestimates the true remaining
cost. With an admissible heuristic, A* is guaranteed to return the
shortest path. If the heuristic overestimates, A* may find a path, but
it might not be the best one.

## Summary
- BFS: complete, optimal, but slow and memory-hungry
- DFS: cheap memory, no optimality guarantee
- A*: optimal with an admissible heuristic, explores far fewer states
