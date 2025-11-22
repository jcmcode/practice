#leetcode problems: 496, 739, 84

def next_greater_elements(arr):
    n = len(arr)
    stack = []
    result = [-1] * n

    for i in range(n):
        while stack and arr[i] > arr[stack[-1]]:
            result[stack.pop()] = arr[i]
        stack.append(i)
    return result    