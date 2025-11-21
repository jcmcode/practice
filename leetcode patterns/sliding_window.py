#Find subarray of size k with maximum sum

#[3,2,7,5,9,6,2] k = 3

#Leetcode Problems for Practice: 643, 3, 76

def max_subarray_sum_sliding_window(arr, k):
    n = len(arr)
    
    window_sum = sum(arr[0:k])
    
    max_sum = window_sum
    max_start_index = 0
    
    for i in range(k, n):
        window_sum = window_sum - arr[i] + arr[i - k]
        if window_sum > max_sum:
            max_sum = window_sum
            max_start_index = i + 1
            
    return arr[max_start_index: max_start_index + k]
            