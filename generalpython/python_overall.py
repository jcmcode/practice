"""
================================================================================
PYTHON OVERALL CHEAT SHEET (PART 1)
================================================================================
A practical, heavily commented reference to core Python. This file is runnable:
- Each section has examples you can uncomment and run.
- Keep everything ASCII for portability.
- Subsequent parts can append more sections (OOP, files, networking, packaging, etc.).

CONTENTS (PART 1):
1) Language Basics: syntax, types, numbers, strings
2) Collections: lists, tuples, sets, dicts, comprehensions
3) Functions: defs, lambdas, args/kwargs, decorators, closures
4) Error Handling: try/except/else/finally, custom exceptions, context managers
5) Iteration Tools: iterators, generators, itertools essentials

Planned for next parts (when you prompt):
- Part 2: OOP, dataclasses, typing, protocols
- Part 3: Files, paths, CSV/JSON, context managers, logging
- Part 4: Concurrency overview (threading/multiprocessing/asyncio) quick hits
- Part 5: Testing (unittest/pytest), packaging, venvs, CLI tooling
================================================================================
"""

import math
import itertools
from collections import defaultdict, Counter, deque
from dataclasses import dataclass

# ============================================================================
# 1. LANGUAGE BASICS
# ============================================================================

def basics_numbers_strings():
    """Show numeric types, string ops, f-strings, slicing"""
    # Numbers
    i = 42                # int (unbounded)
    f = 3.14159           # float (double precision)
    c = 2 + 3j            # complex
    b = 0b1010            # binary literal (10)
    h = 0xFF              # hex literal (255)

    # Math helpers
    root = math.sqrt(16)
    rounded = round(3.14159, 3)

    # Strings
    s = "hello"          # str is Unicode
    multi = """multi\nline"""  # Triple-quoted

    # f-strings
    name = "Ada"
    msg = f"Hi {name}, 2+2={2+2}"

    # Slicing
    text = "abcdefg"
    first3 = text[:3]     # 'abc'
    last2 = text[-2:]     # 'fg'
    step = text[::2]      # 'aceg'

    # Immutability: strings are immutable; concat creates new string
    joined = "-".join(["a", "b", "c"])  # 'a-b-c'

    print(i, f, c, b, h, root, rounded, s, multi, msg, first3, last2, step, joined)

# ============================================================================
# 2. COLLECTIONS
# ============================================================================

def collections_core():
    """Lists, tuples, sets, dicts, comprehension patterns"""
    # Lists (mutable, ordered)
    nums = [1, 2, 3]
    nums.append(4)
    nums.extend([5, 6])
    squared = [x * x for x in nums]           # list comprehension
    evens = [x for x in nums if x % 2 == 0]   # filtered

    # Tuples (immutable, ordered)
    point = (10, 20)
    x, y = point  # unpacking

    # Sets (unique, unordered)
    fruits = {"apple", "banana", "apple"}  # duplicates removed
    fruits.add("cherry")
    has_apple = "apple" in fruits

    # Dicts (key-value mapping)
    user = {"name": "Ada", "age": 36}
    user["role"] = "engineer"
    keys = list(user.keys())
    values = list(user.values())
    items = list(user.items())

    # Dict comprehension
    squares = {n: n * n for n in range(5)}

    # Nested structures
    matrix = [[1, 2], [3, 4]]
    flat = [item for row in matrix for item in row]

    print(nums, squared, evens, point, x, y, fruits, has_apple, user, squares, flat)


def collections_counter_defaultdict_deque():
    """Handy tools from collections"""
    words = "to be or not to be".split()
    freq = Counter(words)  # counts occurrences

    # defaultdict with list factory
    dd = defaultdict(list)
    dd["fruits"].append("apple")
    dd["fruits"].append("banana")

    # deque for fast pops/appends on both ends
    q = deque([1, 2, 3])
    q.appendleft(0)
    q.append(4)
    left = q.popleft()
    right = q.pop()

    print(freq, dd, q, left, right)

# ============================================================================
# 3. FUNCTIONS
# ============================================================================

def functions_args_kwargs():
    """Positional, keyword, *args, **kwargs"""
    def greet(greeting, name, punctuation="!"):
        return f"{greeting}, {name}{punctuation}"

    # *args packs extra positional args into a tuple
    def add_all(*nums):
        return sum(nums)

    # **kwargs packs keyword args into a dict
    def format_kv(**kwargs):
        return ", ".join(f"{k}={v}" for k, v in kwargs.items())

    print(greet("Hi", "Ada"))
    print(add_all(1, 2, 3, 4))
    print(format_kv(alpha=1, beta=2))


def functions_first_class_closures():
    """Functions are first-class; closures capture variables"""
    def make_multiplier(factor):
        def mul(x):
            return x * factor  # factor is captured from outer scope
        return mul

    times3 = make_multiplier(3)
    print(times3(10))  # 30


def functions_lambdas_map_filter_any_all():
    """Lambda expressions and common higher-order helpers"""
    nums = [1, 2, 3, 4, 5]
    doubled = list(map(lambda x: x * 2, nums))
    odds = list(filter(lambda x: x % 2, nums))
    all_positive = all(n > 0 for n in nums)
    any_even = any(n % 2 == 0 for n in nums)
    print(doubled, odds, all_positive, any_even)


def functions_decorators():
    """Simple decorator to log calls"""
    def log_calls(fn):
        def wrapper(*args, **kwargs):
            print(f"Calling {fn.__name__} with {args} {kwargs}")
            result = fn(*args, **kwargs)
            print(f"{fn.__name__} returned {result}")
            return result
        return wrapper

    @log_calls
    def add(a, b):
        return a + b

    add(2, 3)

# ============================================================================
# 4. ERROR HANDLING
# ============================================================================

def error_handling_basics():
    """try/except/else/finally and custom exceptions"""
    class CustomError(Exception):
        pass

    def might_fail(x):
        if x < 0:
            raise CustomError("x must be non-negative")
        return math.sqrt(x)

    try:
        print(might_fail(9))
        print(might_fail(-1))
    except CustomError as e:
        print("Handled custom error:", e)
    except Exception as e:
        print("General exception:", e)
    else:
        print("No errors occurred")
    finally:
        print("This always runs")


def context_manager_example():
    """Use context managers to ensure cleanup"""
    # File context manager handles close automatically
    with open(__file__, "r") as f:
        first_line = f.readline().strip()
        print("First line of this file:", first_line)

# ============================================================================
# 5. ITERATION TOOLS
# ============================================================================

def iterators_and_generators():
    """Create generator functions and use yield"""
    def countdown(n):
        while n > 0:
            yield n
            n -= 1

    print(list(countdown(5)))  # [5, 4, 3, 2, 1]


def itertools_essentials():
    """Handy itertools utilities"""
    nums = [1, 2, 3]
    # accumulate: running totals
    running_sum = list(itertools.accumulate(nums))  # [1, 3, 6]
    # permutations and combinations
    perms = list(itertools.permutations(["a", "b", "c"], 2))
    combos = list(itertools.combinations(["a", "b", "c"], 2))
    # product (cartesian)
    prod = list(itertools.product([1, 2], ["x", "y"]))
    # chain
    chained = list(itertools.chain([1, 2], [3, 4]))
    print(running_sum, perms, combos, prod, chained)


# ============================================================================
# MAIN DEMO (Part 1)
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PYTHON OVERALL CHEAT SHEET (PART 1)")
    print("=" * 70)
    print("Uncomment any function calls below to run demos for that section.\n")

    # 1. Language Basics
    # basics_numbers_strings()

    # 2. Collections
    # collections_core()
    # collections_counter_defaultdict_deque()

    # 3. Functions
    # functions_args_kwargs()
    # functions_first_class_closures()
    # functions_lambdas_map_filter_any_all()
    # functions_decorators()

    # 4. Error Handling
    # error_handling_basics()
    # context_manager_example()

    # 5. Iteration Tools
    # iterators_and_generators()
    # itertools_essentials()

    print("\nNext prompt can add Part 2 (OOP, dataclasses, typing), Part 3 (files/logging), etc.")
    print("=" * 70)
