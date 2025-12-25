#leetcode problems: 102, 994, 127

from collections import deque

def bfs(start_node):
    # 1. Initialize Queue and Visited Set
    queue = deque([start_node])
    visited = set([start_node])
    
    step_count = 0
    
    while queue:
        # Get the number of nodes in the CURRENT layer
        nodes_in_current_layer = len(queue)
        
        # Process strictly this layer
        for _ in range(nodes_in_current_layer):
            current = queue.popleft()
            
            # Check if we reached the target
            if current == target: 
                return step_count
            
            # Add neighbors
            for neighbor in get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # After finishing a full layer, increment distance/step
        step_count += 1