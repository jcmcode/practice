#leetcode problems: 70, 322, 1143, 300, 416, 312

# A Dynamic Programming Implementation (Fibonacci Sequence)

def fib_dynamic(n, memo={}):
    """
    Calculates the nth Fibonacci number using Dynamic Programming (Memoization).
    
    :param n: The position in the sequence to calculate
    :param memo: A dictionary to store results of sub-problems (the "notepad")
    """
    
    # 1. Check the "Notepad" (Base Case for DP)
    # If we have already calculated this number, return the stored result immediately.
    if n in memo:
        return memo[n]
    
    # 2. Base Cases for the logic
    # The first two Fibonacci numbers are 0 and 1.
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    # 3. The Recursive Step with Storage
    # We calculate the value, BUT before returning it, we save it in 'memo'.
    # F(n) = F(n-1) + F(n-2)
    result = fib_dynamic(n - 1, memo) + fib_dynamic(n - 2, memo)
    
    # Store the result in our dictionary so we never have to calculate 'n' again
    memo[n] = result
    
    return result

# --- Driver Code ---
if __name__ == "__main__":
    import time
    
    # Let's try to calculate the 50th Fibonacci number.
    # Without DP, this would take a very long time (millions of operations).
    # With DP, it happens instantly.
    
    target_n = 50
    
    print(f"Calculating Fibonacci number at position {target_n}...")
    print("-" * 40)
    
    start_time = time.time()
    answer = fib_dynamic(target_n)
    end_time = time.time()
    
    print(f"Result: {answer}")
    print(f"Time taken: {end_time - start_time:.6f} seconds")
    print("-" * 40)
    
    # To prove it worked, let's print a few from the dictionary
    # Note: The dictionary persists because it is a mutable default argument in Python 
    # (a quirk used here for simplicity, though often avoided in production for class methods).
    print("Some stored values in the memo/notepad:")
    keys_to_show = [5, 10, 15]
    # We have to access the memo from a fresh call or if we had passed it externally.
    # For demonstration, we will just run small calls to see they return instantly.
    print(f"Fib(10) re-check: {fib_dynamic(10)}")