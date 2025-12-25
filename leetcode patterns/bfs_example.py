from collections import deque
from typing import List, Optional

# ==========================================
# 1. HELPER CLASS (Binary Tree Node)
# ==========================================
# This definition is standard for all LeetCode tree problems.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# ==========================================
# 2. BFS SOLUTION (Level Order Traversal)
# ==========================================
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # Edge Case: If tree is empty, return empty list
        if not root:
            return []
        
        results = []
        
        # Initialize Queue with the root
        queue = deque([root])
        
        # While there are nodes to process...
        while queue:
            level_values = []
            
            # SNAPSHOT: How many nodes are in this specific level?
            # We must fix this size because we will be appending new 
            # children to the queue inside the loop, but those children 
            # belong to the NEXT level.
            level_size = len(queue)
            
            for _ in range(level_size):
                # 1. Remove node from the front (First In, First Out)
                node = queue.popleft()
                
                # 2. Process the node (add to current level list)
                level_values.append(node.val)
                
                # 3. Add children to the back of the queue
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            # Level is complete, add to main results
            results.append(level_values)
            
        return results

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Let's manually build a tree: [3, 9, 20, null, null, 15, 7]
    #      3
    #     / \
    #    9  20
    #      /  \
    #     15   7
    
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)

    solver = Solution()
    print("BFS Level Order Output:", solver.levelOrder(root))
    # Expected Output: [[3], [9, 20], [15, 7]]