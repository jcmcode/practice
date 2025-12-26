#leetcode problems: 133, 113, 210


# A Depth First Search (DFS) Implementation in Python

def dfs_recursive(graph, current_node, visited=None):
    """
    Traverses a graph using Depth First Search.
    
    :param graph: A dictionary representing the adjacency list of the graph
    :param current_node: The starting node for this particular step
    :param visited: A set to keep track of visited nodes (avoids infinite loops)
    """
    
    # 1. Initialize the visited set if this is the first call
    if visited is None:
        visited = set()
    
    # 2. Mark the current node as visited and process it (e.g., print it)
    visited.add(current_node)
    print(f"Visited: {current_node}")
    
    # 3. Explore neighbors
    # We look at every neighbor of the current node
    for neighbor in graph[current_node]:
        
        # 4. Recursive Step
        # If the neighbor hasn't been visited yet, we 'dive' into it immediately.
        # This is what makes it "Depth First" - we go deeper before checking other siblings.
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
            
    return visited

# --- Driver Code (This sets up the graph and runs the function) ---
if __name__ == "__main__":
    # We represent the graph using a dictionary (Adjacency List).
    # 'A': ['B', 'C'] means Node A is connected to Node B and Node C.
    my_graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }

    print("Starting DFS traversal from Node A:")
    print("-" * 30)
    
    # Start the search
    dfs_recursive(my_graph, 'A')
    
    print("-" * 30)
    print("Traversal Complete.")