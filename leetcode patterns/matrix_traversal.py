#leetcode problems: 733, 200, 130

# Matrix Traversal using DFS (Depth First Search)

def traverse_matrix(grid, row, col, visited):
    """
    Recursively visits all connected cells in a matrix starting from (row, col).
    
    :param grid: List of Lists representing the 2D matrix
    :param row: Current row index
    :param col: Current column index
    :param visited: A set containing tuples of (row, col) that we have seen
    """
    
    # 1. Calculate grid dimensions
    rows_max = len(grid)
    cols_max = len(grid[0])
    
    # 2. BASE CASES (The "Stop" Conditions)
    # Check bounds: Are we off the grid?
    if row < 0 or row >= rows_max or col < 0 or col >= cols_max:
        return
    
    # Check visited: Have we been here before?
    if (row, col) in visited:
        return
    
    # Check content: (Optional) Assume we only want to traverse '1's (Land)
    # If we hit a '0' (Water), we stop this path.
    if grid[row][col] == 0:
        return

    # 3. Process the cell
    visited.add((row, col))
    print(f"Visiting cell at ({row}, {col}) -> Value: {grid[row][col]}")
    
    # 4. Define Directions (Up, Down, Left, Right)
    # Each item is a change in (row, col)
    directions = [
        (-1, 0), # Up
        (1, 0),  # Down
        (0, -1), # Left
        (0, 1)   # Right
    ]
    
    # 5. Recursive Step: Visit all neighbors
    for dr, dc in directions:
        new_row = row + dr
        new_col = col + dc
        traverse_matrix(grid, new_row, new_col, visited)

# --- Driver Code ---
if __name__ == "__main__":
    # 1 represents Land (path), 0 represents Water (wall)
    my_grid = [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1]
    ]
    
    print("Grid Structure:")
    for r in my_grid:
        print(r)
    print("-" * 30)
    
    visited_set = set()
    
    # Start traversal from top-left (0,0)
    print("Starting Traversal from (0,0):")
    traverse_matrix(my_grid, 0, 0, visited_set)
    
    print("-" * 30)
    # Notice that the '1' at (3,0) and (3,3) might not be visited 
    # if they are not connected to the start point!
    print(f"Total cells visited: {len(visited_set)}")