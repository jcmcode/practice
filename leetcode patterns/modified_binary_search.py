#leetcode problems: 33, 153, 240

# Modified Binary Search (Rotated Array Search)

def search_rotated_array(nums, target):
    """
    Searches for a target value in a rotated sorted array.
    
    :param nums: List of integers (rotated sorted)
    :param target: Integer to find
    :return: Index of target, or -1 if not found
    """
    left = 0
    right = len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        # 1. Found it?
        if nums[mid] == target:
            return mid
        
        # 2. Identify which half is sorted
        
        # CASE A: Left side is sorted
        # We know this because the start is less than the middle
        if nums[left] <= nums[mid]:
            print(f"Checking Left Half: [{nums[left]} ... {nums[mid]}] is sorted.")
            
            # Is our target inside this sorted range?
            if nums[left] <= target < nums[mid]:
                right = mid - 1  # Target is in the left half
            else:
                left = mid + 1   # Target is in the right (unsorted) half
                
        # CASE B: Right side is sorted
        # If left wasn't sorted, right MUST be sorted
        else:
            print(f"Checking Right Half: [{nums[mid]} ... {nums[right]}] is sorted.")
            
            # Is our target inside this sorted range?
            if nums[mid] < target <= nums[right]:
                left = mid + 1   # Target is in the right half
            else:
                right = mid - 1  # Target is in the left (unsorted) half

    return -1

# --- Driver Code ---
if __name__ == "__main__":
    # Example: Sorted array [0, 1, 2, 3, 4, 5, 6, 7]
    # Rotated at pivot index 4 -> [4, 5, 6, 7, 0, 1, 2, 3]
    rotated_list = [4, 5, 6, 7, 0, 1, 2, 3]
    target_val = 1
    
    print(f"List: {rotated_list}")
    print(f"Target: {target_val}")
    print("-" * 40)
    
    result_index = search_rotated_array(rotated_list, target_val)
    
    print("-" * 40)
    if result_index != -1:
        print(f"Found target {target_val} at index: {result_index}")
    else:
        print("Target not found.")