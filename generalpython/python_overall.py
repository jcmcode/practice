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
CONTENTS (PART 1 + PART 2 + PART 3):
1) Language Basics: syntax, types, numbers, strings
2) Collections: lists, tuples, sets, dicts, comprehensions
3) Functions: defs, lambdas, args/kwargs, decorators, closures
4) Error Handling: try/except/else/finally, custom exceptions, context managers
5) Iteration Tools: iterators, generators, itertools essentials
6) OOP Essentials: classes, instances, attributes, class/static methods
7) Inheritance & Composition: super(), mixins, composition vs inheritance
8) Dataclasses: auto-__init__, defaults, slots, immutability (frozen)
9) Typing & Protocols: type hints, TypedDict, Protocol, Generics
10) Files & Paths: pathlib, reading/writing text & binary
11) CSV & JSON: csv module, json module, newline handling
12) Logging Basics: logging setup, levels, handlers

Planned for next parts (when you prompt):
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
# 6. OOP ESSENTIALS
# ============================================================================

def oop_basics():
    """Classes, instances, methods, class attrs, static/class methods"""

    class Animal:
        # Class attribute (shared)
        kingdom = "animalia"

        def __init__(self, name, species):
            self.name = name          # Instance attribute
            self.species = species

        def speak(self):
            # Instance method: first arg is self
            return f"{self.name} makes a sound"

        @classmethod
        def kingdom_name(cls):
            # Class method: first arg is class (cls)
            return cls.kingdom

        @staticmethod
        def is_alive():
            # Static method: no auto self/cls, utility-like
            return True

    dog = Animal("Fido", "canine")
    cat = Animal("Whiskers", "feline")

    print(dog.speak())
    print(cat.speak())
    print(Animal.kingdom_name())
    print(Animal.is_alive())


# ============================================================================
# 7. INHERITANCE & COMPOSITION
# ============================================================================

def inheritance_and_composition():
    """super(), mixins, composition vs inheritance"""

    class Vehicle:
        def __init__(self, make):
            self.make = make

        def move(self):
            return "Moving"

    class Car(Vehicle):
        def __init__(self, make, doors):
            super().__init__(make)  # call parent init
            self.doors = doors

        def move(self):
            base = super().move()
            return f"{base} on the road"

    # Mixin example (adds behavior but not a full type on its own)
    class ElectricMixin:
        def charge(self):
            return "Charging battery"

    class ElectricCar(ElectricMixin, Car):
        pass

    # Composition example: has-a relationship
    class Engine:
        def start(self):
            return "Engine started"

    class Boat:
        def __init__(self):
            self.engine = Engine()  # composed object

        def start(self):
            return self.engine.start()

    car = ElectricCar("Tesla", 4)
    boat = Boat()

    print(car.move(), car.charge())
    print(boat.start())


# ============================================================================
# 8. DATA CLASSES
# ============================================================================

def dataclasses_examples():
    """Auto-__init__, defaults, ordering, immutability"""

    @dataclass
    class Point:
        x: float
        y: float

    @dataclass(order=True)
    class Task:
        priority: int
        description: str
        done: bool = False  # default value

    @dataclass(frozen=True)
    class Config:
        debug: bool
        version: str = "1.0"

    @dataclass(slots=True)
    class User:
        name: str
        email: str

    p = Point(1, 2)
    t1 = Task(1, "write code")
    t2 = Task(2, "review PR", done=True)
    cfg = Config(debug=False)
    usr = User("Ada", "ada@example.com")

    print(p, t1, t2)
    print(cfg, cfg.version)
    print(usr.name)


# ============================================================================
# 9. TYPING & PROTOCOLS
# ============================================================================

def typing_and_protocols():
    """Type hints, TypedDict, Protocol, Generics"""
    from typing import TypedDict, Protocol, List, Dict, Optional, Iterable, TypeVar, Generic

    class UserDict(TypedDict):
        id: int
        name: str
        email: str
        active: bool

    # Protocol: structural typing (duck typing with types)
    class Greeter(Protocol):
        def greet(self, name: str) -> str:
            ...

    class FriendlyBot:
        def greet(self, name: str) -> str:
            return f"Hello {name}!"

    # Generic container example
    T = TypeVar("T")

    class Box(Generic(T)):
        def __init__(self, value: T):
            self.value = value

        def get(self) -> T:
            return self.value

    def greet_all(greeter: Greeter, names: Iterable[str]) -> List[str]:
        return [greeter.greet(n) for n in names]

    user: UserDict = {"id": 1, "name": "Ada", "email": "ada@ex.com", "active": True}
    bot = FriendlyBot()
    box_int = Box(123)
    box_str = Box("hello")

    greetings = greet_all(bot, ["Bob", "Carol"])

    print(user)
    print(box_int.get(), box_str.get())
    print(greetings)


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

    # 6. OOP Essentials
    # oop_basics()

    # 7. Inheritance & Composition
    # inheritance_and_composition()

    # 8. Dataclasses
    # dataclasses_examples()

    # 9. Typing & Protocols
    # typing_and_protocols()

    print("\nNext prompt can add Part 2 (OOP, dataclasses, typing), Part 3 (files/logging), etc.")
    print("=" * 70)
