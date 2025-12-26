#leetcode problems: 56, 57, 435

# Overlapping Intervals Implementation

def merge_intervals(intervals):
    """
    Merges overlapping intervals.
    
    :param intervals: A list of lists, where each inner list is [start, end]
    :return: A list of merged intervals
    """
    
    # 1. Edge Case: Empty list
    if not intervals:
        return []

    # 2. SORTING (The most important step)
    # We sort based on the first element (start time) of each interval.
    # key=lambda x: x[0] tells Python to look at index 0 for sorting.
    intervals.sort(key=lambda x: x[0])
    
    # Initialize the result list with the first interval
    merged = [intervals[0]]
    
    # 3. Iterate through the rest of the intervals
    for current in intervals[1:]:
        
        # Get the last interval we added to the 'merged' list
        last_added = merged[-1]
        
        # 4. Check for Overlap
        # Current Start Time <= Last Added End Time
        if current[0] <= last_added[1]:
            # MERGE ACTION:
            # We don't add a new interval. We just update the end time of the 
            # existing one. We take the max because one might be fully inside the other.
            last_added[1] = max(last_added[1], current[1])
            print(f"Merged: {last_added} with {current} -> New End: {last_added[1]}")
            
        else:
            # NO OVERLAP:
            # The current interval starts after the last one ends.
            # So we just add it to our list.
            merged.append(current)
            print(f"No Overlap: Added {current}")
            
    return merged

# --- Driver Code ---
if __name__ == "__main__":
    # Example input: Unsorted and overlapping
    # [1, 3] and [2, 6] overlap
    # [8, 10] is standalone
    # [15, 18] is standalone
    input_intervals = [[1, 3], [8, 10], [2, 6], [15, 18]]
    
    print(f"Original Intervals: {input_intervals}")
    print("-" * 40)
    
    result = merge_intervals(input_intervals)
    
    print("-" * 40)
    print(f"Final Merged Intervals: {result}")