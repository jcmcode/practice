#leetcode problems: 46, 78, 51

#generally O(N!) or O(2^N) and used for small input sized problems

import copy
from typing import List

def backtracking_template(candidates):
    """
    This is a skeleton function. It uses placeholders (like is_solution)
    so it compiles without errors, but you must replace logic to use it.
    """
    output = []
    
    def is_solution(state):
        # TODO: Check if state is a complete valid solution
        return False 

    def is_valid(candidate, state):
        # TODO: Check if candidate can be added to current state
        return True

    def backtrack(current_state):
        # 1. Base Case: Have we reached a valid solution?
        if is_solution(current_state):
            output.append(current_state[:]) # Important: Make a copy!
            return

        # 2. Iterate through candidates for the next step
        for candidate in candidates:
            if is_valid(candidate, current_state):
                # A. Choose
                current_state.append(candidate)
                
                # B. Explore (recurse)
                backtrack(current_state)
                
                # C. Un-choose (backtrack)
                current_state.pop()

    # Start with an empty state
    backtrack([])
    return output