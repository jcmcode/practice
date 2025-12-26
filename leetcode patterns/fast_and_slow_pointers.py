#Fast and slow pointer pattern

#Used to find cycles in a linked list or array

#move one pointer at normal speed and the other at double speed

#Leetcode problems that use this pattern: 141, 202, 287

# Fast and Slow Pointer Implementation (Cycle Detection)

class ListNode:
    """A helper class to create nodes for our linked list."""
    def __init__(self, value):
        self.value = value
        self.next = None

def has_cycle(head):
    """
    Detects if a linked list has a cycle using Fast & Slow pointers.
    
    :param head: The first node of the linked list
    :return: True if a cycle exists, False otherwise
    """
    
    # Initialize both pointers at the start
    slow_ptr = head
    fast_ptr = head
    
    # We continue as long as the Fast pointer hasn't fallen off the end.
    # We must check 'fast_ptr.next' to ensure we can jump two steps safely.
    while fast_ptr is not None and fast_ptr.next is not None:
        
        # 1. Move Slow pointer one step
        slow_ptr = slow_ptr.next
        
        # 2. Move Fast pointer two steps
        fast_ptr = fast_ptr.next.next
        
        # 3. Check for collision
        # If they are pointing to the exact same object in memory, we found a loop.
        if slow_ptr == fast_ptr:
            return True
            
    # If the loop finishes, it means Fast reached the end (None).
    # Therefore, the list is straight (no cycle).
    return False

# --- Driver Code ---
if __name__ == "__main__":
    # 1. Create Nodes
    node1 = ListNode(1)
    node2 = ListNode(2)
    node3 = ListNode(3)
    node4 = ListNode(4)
    
    # 2. Link them to form: 1 -> 2 -> 3 -> 4
    node1.next = node2
    node2.next = node3
    node3.next = node4
    
    # Test Case 1: No Cycle
    print("Test 1 (Straight List): Is there a cycle?", has_cycle(node1))
    
    # 3. Create a Cycle
    # We point the last node (4) back to the second node (2)
    # 1 -> 2 -> 3 -> 4
    #      ^---------|
    node4.next = node2
    
    # Test Case 2: With Cycle
    print("Test 2 (Looped List):   Is there a cycle?", has_cycle(node1))