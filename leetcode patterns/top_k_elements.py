#k largest elements use min heap

#k smallest elements use max heap

#leetcode problems: 215, 347, 373

import heapq

def find_k_largest(nums, k):
    """
    Finds the K largest elements using a Min Heap.
    Time Complexity: O(N log K)
    """
    # 1. Create a Min Heap with the first K elements
    # We slice the list [:k] to get the first k items
    min_heap = nums[:k]
    
    # Transform list into a heap in-place (O(k))
    heapq.heapify(min_heap)
    
    # 2. Iterate through the REST of the numbers
    for num in nums[k:]:
        
        # 3. The Comparison
        # Look at the smallest item in the heap (heap[0]).
        # If the current number is BIGGER than the smallest "VIP",
        # we kick out the small one and add the big one.
        if num > min_heap[0]:
            heapq.heapreplace(min_heap, num)
            # Note: heapreplace is faster than heappop() followed by heappush()
            
    return min_heap


def find_k_smallest(nums, k):
    """
    Finds the K smallest elements using a Max Heap (simulated with negatives).
    Time Complexity: O(N log K)
    """
    # 1. Create a "Max Heap" with the first K elements
    # We negate the numbers so that the largest number becomes the smallest
    # (e.g., 10 becomes -10, 2 becomes -2. -10 is smaller than -2 in Python's eyes)
    max_heap = [-n for n in nums[:k]]
    
    heapq.heapify(max_heap)
    
    # 2. Iterate through the rest
    for num in nums[k:]:
        
        # We negate the current number to match our heap's logic
        val = -num
        
        # 3. The Comparison
        # In our inverted world, heap[0] is the "smallest" negative number
        # which corresponds to the LARGEST actual number in our collection.
        # If our new inverted number is LARGER (meaning closer to 0, actually smaller),
        # we swap.
        
        # Example: Heap has [-100], val is -5. 
        # -5 > -100 is True. So we replace -100 with -5.
        # We just swapped 100 (big) for 5 (small). Correct!
        if val > max_heap[0]:
            heapq.heapreplace(max_heap, val)
            
    # 4. Flip the signs back to positive before returning
    return [-n for n in max_heap]


# --- Driver Code ---
if __name__ == "__main__":
    # Example Data
    data = [3, 2, 1, 5, 6, 4, 10, 9, 8, 7]
    k_value = 3
    
    print(f"Original Data: {data}")
    print(f"Finding top {k_value} elements...")
    print("-" * 40)
    
    # Test K Largest
    largest = find_k_largest(data, k_value)
    # The heap doesn't guarantee sorted order, just that the items are the top K.
    # We sort them here just for a pretty print output.
    print(f"Top {k_value} Largest: {sorted(largest)}") 
    
    # Test K Smallest
    smallest = find_k_smallest(data, k_value)
    print(f"Top {k_value} Smallest: {sorted(smallest)}")