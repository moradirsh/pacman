# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    # FOR DEPTH FIRST SEARCH: USE STACK

    # Stack for Depth First Search is Last In, First Out (LIFO) util.py has Stack and Queue classes
    stack = util.Stack()
    # Track nodes to avoid cycles
    visited = set()
    # Stack stores start state in the data structure with the path
    stack.push((problem.getStartState(), []))
    
    while not stack.isEmpty():
        # Pop the most recently added state and add to path / make current_state(DFS)
        current_state, path = stack.pop()
        # Skip if visited
        if current_state in visited:
            continue # Loop to next iteration
        # Mark current state visited
        visited.add(current_state)
        # Check if at goal state
        if problem.isGoalState(current_state):
            return path
        # Add all children to the stack
        for successor, action, stepCost in problem.getSuccessors(current_state):
            if successor not in visited:
                # If the successor has not been visited, create new path by adding current action
                new_path = path + [action]
                stack.push((successor, new_path))
                
    # Return empty list because no solution would have been found
    return []

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""

    # FOR BREADTH FIRST SEARCH: USE QUEUE

    # Queue for Breadth First Search is First In, First Out (FIFO)
    queue = util.Queue()
    # Track nodes to avoid cycles
    visited = set()
    # Queue stores start state in the data structure with the path
    queue.push((problem.getStartState(), []))
    
    while not queue.isEmpty():
        # Pop the oldest added state and add to path / make current_state (BFS)
        current_state, path = queue.pop()
        # Skip if visited
        if current_state in visited:
            continue # Loop to next iteration
        # Mark current state visited
        visited.add(current_state)

        # Check if at goal state
        if problem.isGoalState(current_state):
            return path

        # Add all children to the queue
        for successor, action, stepCost in problem.getSuccessors(current_state):
            if successor not in visited:
                # If the successor has not been visited, create new path by adding current action
                new_path = path + [action]
                queue.push((successor, new_path))
                
    # Return empty list because no solution would have been found
    return []

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    
    # FOR UNIFORM COST SEARCH: USE PRIORITY QUEUE
    # optimal solutions by always expanding the lowest-cost node first
    
    # Priority Queue for Uniform Cost Search orders by path cost (lower cost = higher priority)
    priority_queue = util.PriorityQueue()
    
    # Track visited states to avoid redundant exploration
    visited = set()
    # Priority queue stores start state with path and cost ((state, path), cost)
    priority_queue.push((problem.getStartState(), []), 0)
    
    while not priority_queue.isEmpty():
        # Pop state with lowest accumulated cost
        current_state, path = priority_queue.pop()
        # Skip if visited
        if current_state in visited:
            continue # Loop to next iteration
        # Mark current state visited
        visited.add(current_state)
        
        # Check if at goal state
        if problem.isGoalState(current_state):
            return path
        
        # Add all children to the priority queue
        for successor, action, stepCost in problem.getSuccessors(current_state):
            if successor not in visited:
                # If the successor has not been visited, create new path by adding current action
                new_path = path + [action]
                
                # Calculate total cost from start to successor:
                # Cost of current path + cost of this step
                new_cost = problem.getCostOfActions(path) + stepCost
                
                # Add successor to queue with its path and total cost as priority
                priority_queue.push((successor, new_path), new_cost)
                
    # Return empty list because no solution would have been found
    return []

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    
    # FOR A* SEARCH: USE PRIORITY QUEUE WITH f(n) = g(n) + h(n)
    # f(n) = g(n) + h(n) where (lower f = higher priority):
    # g(n) = actual cost from start to current node
    # h(n) = heuristic estimate from current node to goal
    # f(n) = estimated total cost from start to goal through current node
    priority_queue = util.PriorityQueue()
    
    # Use dictionary instead of set to track best cost to reach each state
    # This allows revisiting states if we find a better (cheaper) path
    visited = {}
    
    # Get starting state and calculate initial heuristic
    start_state = problem.getStartState()
    start_h_cost = heuristic(start_state, problem)
    # Start has g_cost=0, so f_cost = 0 + h(start) = h(start)
    priority_queue.push((start_state, [], 0), start_h_cost)
    
    while not priority_queue.isEmpty():
        # Pop state with lowest f(n) = g(n) + h(n) value
        # Extract: current state, path to reach it, and actual cost g(n)
        current_state, path, g_cost = priority_queue.pop()
        
        # Skip if we've reached this state before with equal or better cost (allows finding better paths to same state)
        if current_state in visited and visited[current_state] <= g_cost:
            continue # Jump to next iteration of while loop
            
        # Record this as the best known cost to reach this state
        visited[current_state] = g_cost
        
        # Check if at goal state
        if problem.isGoalState(current_state):
            return path
        
        # Add all children to the priority queue
        for successor, action, stepCost in problem.getSuccessors(current_state):
            # Calculate new g(n) for successor: current cost + step cost
            new_g_cost = g_cost + stepCost
            
            # Only add successor if:
            # We haven't seen this state before, OR We found a better (lower cost) path to this state
            if successor not in visited or visited[successor] > new_g_cost:
                # Create path to successor by appending action to current path
                new_path = path + [action]
                # Calculate h(n): heuristic estimate from successor to goal
                h_cost = heuristic(successor, problem)
                # Calculate f(n) = g(n) + h(n): total estimated cost
                f_cost = new_g_cost + h_cost
                
                # Add successor to queue: (state, path, g_cost) with f_cost 
                priority_queue.push((successor, new_path, new_g_cost), f_cost)
                
    # Return empty list because no solution would have been found
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
