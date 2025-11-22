#Leetcode Problems for Practice: 303,525,560

# Array = [1,2,3,4,5,6,7]
# Prefix Sum Array (PFS) = [1,3,6,10,15,21,28]

# Sum = PFS[i,j] = PFS[j] - PFS[i-1]

def create_prefix_sum(arr):
    for i in range(1, len(arr)):
        arr[i] += arr[i-1]
    return arr