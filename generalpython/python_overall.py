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
CONTENTS (PART 1 + PART 2 + PART 3 + PART 4 + PART 5 + EXTRAS):
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
13) Concurrency Quick Hits: threading, ThreadPoolExecutor, multiprocessing, asyncio
14) Testing: unittest basics, pytest-style examples, fixtures mindset
15) Packaging & Environments: venv basics, minimal package layout, CLI entrypoints
16) CLI (argparse) quick start
17) Datetime & timezones (zoneinfo)
18) Regex essentials
19) HTTP basics (urllib + requests-style)
20) SQLite mini demo
21) contextlib utilities
22) tempfile quick hits
23) Profiling & debugging (timeit, cProfile, breakpoint)
24) Asyncio advanced patterns (tasks, semaphore, cancellation)
25) Testing extras (unittest.mock, parametrization mindset)

Planned for next parts (when you prompt):
- (Completed through 25 sections)
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
# 10. FILES & PATHS (pathlib)
# ============================================================================

def files_and_paths():
    """Use pathlib for clean, cross-platform file handling"""
    from pathlib import Path

    base = Path(__file__).parent  # directory containing this file
    data_dir = base / "data"      # construct paths with /
    sample_file = data_dir / "sample.txt"

    # Ensure directory exists
    data_dir.mkdir(exist_ok=True)

    # Write text (overwrites)
    sample_file.write_text("hello world\nsecond line", encoding="utf-8")

    # Read text
    content = sample_file.read_text(encoding="utf-8")

    # Append text
    with sample_file.open("a", encoding="utf-8") as f:
        f.write("\nappended line")

    # Binary write/read
    bin_file = data_dir / "bytes.bin"
    bin_file.write_bytes(b"\x00\x01\x02")
    raw = bin_file.read_bytes()

    # Listing directory
    files = [p.name for p in data_dir.iterdir() if p.is_file()]

    print("Content:\n", content)
    print("Raw bytes:", raw)
    print("Files in data/:", files)


# ============================================================================
# 11. CSV & JSON
# ============================================================================

def csv_and_json_examples():
    """CSV read/write, JSON read/write"""
    import csv, json
    from pathlib import Path

    base = Path(__file__).parent
    data_dir = base / "data"
    data_dir.mkdir(exist_ok=True)

    # Sample data
    rows = [
        {"id": 1, "name": "Ada", "lang": "Python"},
        {"id": 2, "name": "Linus", "lang": "C"},
    ]

    # --- CSV write ---
    csv_path = data_dir / "people.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "lang"])
        writer.writeheader()
        writer.writerows(rows)

    # --- CSV read ---
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        loaded_csv = list(reader)

    # --- JSON write ---
    json_path = data_dir / "people.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # --- JSON read ---
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))

    print("CSV loaded:", loaded_csv)
    print("JSON loaded:", loaded_json)


# ============================================================================
# 12. LOGGING BASICS
# ============================================================================

def logging_basics():
    """Set up basic logging with levels and handlers"""
    import logging
    from pathlib import Path

    log_path = Path(__file__).parent / "data" / "app.log"
    log_path.parent.mkdir(exist_ok=True)

    # BasicConfig should be called once at startup
    logging.basicConfig(
        level=logging.INFO,  # DEBUG < INFO < WARNING < ERROR < CRITICAL
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger("demo")
    logger.debug("This is debug (won't show at INFO level)")
    logger.info("Informational message")
    logger.warning("Something to be aware of")
    logger.error("An error occurred")

    print(f"Logs written to {log_path}")


# ============================================================================
# 13. CONCURRENCY QUICK HITS (threading / multiprocessing / asyncio)
# ============================================================================

def threading_quick_demo():
    """Threading for I/O-bound tasks (concurrent but not parallel CPU)"""
    import threading, time

    def io_task(name, duration):
        print(f"{name} start")
        time.sleep(duration)  # simulate I/O wait
        print(f"{name} done")

    t1 = threading.Thread(target=io_task, args=("T1", 1.0))
    t2 = threading.Thread(target=io_task, args=("T2", 1.0))

    t1.start(); t2.start()
    t1.join(); t2.join()
    print("Threads finished")


def threadpool_quick_demo():
    """ThreadPoolExecutor for simple parallel I/O tasks"""
    import concurrent.futures, time

    def fetch(n):
        time.sleep(0.5)
        return f"result-{n}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for result in ex.map(fetch, range(5)):
            print(result)


def multiprocessing_quick_demo():
    """Multiprocessing for CPU-bound tasks (true parallelism)"""
    import multiprocessing, math

    def cpu_task(n):
        return sum(int(math.sqrt(i)) for i in range(n))

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(cpu_task, [200_000, 200_000, 200_000, 200_000])
    print("Multiprocessing results:", results)


async def asyncio_quick_demo():
    """Asyncio for many concurrent I/O tasks in one thread"""
    import asyncio

    async def io_op(name, delay):
        print(f"{name} start")
        await asyncio.sleep(delay)
        print(f"{name} done")
        return name

    results = await asyncio.gather(
        io_op("A", 1.0),
        io_op("B", 0.5),
        io_op("C", 0.2),
    )
    print("Asyncio results:", results)


def run_asyncio_quick_demo():
    import asyncio
    asyncio.run(asyncio_quick_demo())


# ============================================================================
# 14. TESTING (unittest + pytest style)
# ============================================================================

def testing_unittest_basics():
    """Demonstrate a small unittest TestCase"""
    import unittest

    class MathTests(unittest.TestCase):
        def test_add(self):
            self.assertEqual(2 + 2, 4)

        def test_raises(self):
            with self.assertRaises(ZeroDivisionError):
                _ = 1 / 0

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MathTests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def testing_pytest_style_examples():
    """Pytest-like patterns (no actual pytest dependency needed to read)"""

    def add(a, b):
        return a + b

    def test_add_simple():
        assert add(2, 3) == 5

    def test_division_by_zero():
        try:
            _ = 1 / 0
        except ZeroDivisionError:
            pass
        else:
            raise AssertionError("Expected ZeroDivisionError")

    # Run "tests" manually here (in pytest they are auto-discovered)
    test_add_simple()
    test_division_by_zero()
    print("Pytest-style checks passed (manual run)")


# ============================================================================
# 15. PACKAGING & ENVIRONMENTS
# ============================================================================

def packaging_and_venv_notes():
    """Minimal notes on venv, package layout, CLI entrypoints"""

    notes = r"""
VENV (BUILT-IN):
----------------
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip

MINIMAL PACKAGE LAYOUT:
-----------------------
project/
  pyproject.toml        # build config (recommended)
  README.md
  src/
    mypkg/
      __init__.py
      core.py
  tests/
    test_core.py

PYPROJECT.TOML (POETRY/PEP 621 STYLE EXAMPLE):
---------------------------------------------
[project]
name = "mypkg"
version = "0.1.0"
description = "My sample package"
requires-python = ">=3.10"
dependencies = [
  "requests>=2.31",
]

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

CLI ENTRYPOINT (SETUPTOOLS-STYLE):
----------------------------------
In pyproject.toml under [project.scripts]:
mycli = "mypkg.core:main"

Then implement in src/mypkg/core.py:
def main():
    print("Hello from CLI")

INSTALL LOCALLY (editable):
---------------------------
pip install -e .

RUN TESTS:
----------
pytest

"""
    print(notes)


# ============================================================================
# 16. CLI (argparse) QUICK START
# ============================================================================

def argparse_quick_start():
    """Argparse demo using a fake argv list so it is safe to run here."""
    import argparse

    parser = argparse.ArgumentParser(description="Demo CLI")
    parser.add_argument("name", help="Name to greet")
    parser.add_argument("--count", type=int, default=1, help="How many times")
    parser.add_argument("--loud", action="store_true", help="Uppercase output")

    args = parser.parse_args(["Ada", "--count", "2", "--loud"])  # fake argv

    greeting = "hello" if not args.loud else "HELLO"
    for _ in range(args.count):
        print(f"{greeting}, {args.name}!")


# ============================================================================
# 17. DATETIME & TIMEZONES
# ============================================================================

def datetime_and_timezones():
    """Working with datetime, parsing, formatting, zoneinfo"""
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    now = datetime.now(tz=timezone.utc)
    later = now + timedelta(hours=3)

    ny = now.astimezone(ZoneInfo("America/New_York"))
    berlin = now.astimezone(ZoneInfo("Europe/Berlin"))

    parsed = datetime.strptime("2024-12-31 23:30", "%Y-%m-%d %H:%M")
    formatted = parsed.strftime("%b %d, %Y %I:%M %p")

    print("UTC now:", now.isoformat())
    print("UTC later:", later.isoformat())
    print("NY:", ny.isoformat())
    print("Berlin:", berlin.isoformat())
    print("Parsed:", parsed)
    print("Formatted:", formatted)


# ============================================================================
# 18. REGEX ESSENTIALS
# ============================================================================

def regex_essentials():
    """Compile patterns, use groups, findall/search/sub"""
    import re

    text = "Contact us at support@example.com or sales@example.org"
    email_pattern = re.compile(r"(?P<user>[\w.-]+)@(?P<host>[\w.-]+)")

    all_emails = email_pattern.findall(text)
    first = email_pattern.search(text)
    redacted = email_pattern.sub("<hidden>", text)

    print("Emails:", all_emails)
    if first:
        print("First user:", first.group("user"))
        print("First host:", first.group("host"))
    print("Redacted:", redacted)


# ============================================================================
# 19. HTTP BASICS (urllib + requests-style)
# ============================================================================

def http_basics():
    """Tiny HTTP examples; handles offline gracefully."""
    import urllib.request

    try:
        with urllib.request.urlopen("https://example.com", timeout=3) as resp:
            body = resp.read(80).decode("utf-8", errors="replace")
            print("urllib status:", resp.status)
            print("Body preview:", body.replace("\n", " ")[:120])
    except Exception as e:
        print("urllib fetch skipped or failed:", e)

    try:
        import requests  # type: ignore
    except Exception:
        requests = None

    if requests:
        try:
            r = requests.get("https://example.com", timeout=3)
            print("requests status:", r.status_code)
            print("requests text preview:", r.text[:120].replace("\n", " "))
        except Exception as e:
            print("requests fetch failed:", e)
    else:
        print("requests not installed; pip install requests>=2.31 if needed")


# ============================================================================
# 20. SQLITE MINI DEMO
# ============================================================================

def sqlite_mini_demo():
    """Create table, insert, query in memory."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    cur.executemany("INSERT INTO users (name) VALUES (?)", [("Ada",), ("Linus",)])
    conn.commit()

    cur.execute("SELECT id, name FROM users ORDER BY id")
    rows = cur.fetchall()
    print("Rows:", rows)

    conn.close()


# ============================================================================
# 21. CONTEXTLIB UTILITIES
# ============================================================================

def contextlib_utilities():
    """Show @contextmanager, ExitStack, and suppress."""
    import contextlib, time, io

    @contextlib.contextmanager
    def timer(label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            print(f"{label} took {elapsed:.4f}s")

    with timer("sleep 0.1"):
        time.sleep(0.1)

    # ExitStack to manage multiple contexts dynamically
    with contextlib.ExitStack() as stack:
        buf = stack.enter_context(io.StringIO())
        buf.write("hello exitstack")
        print("ExitStack buffer:", buf.getvalue())

    # suppress to ignore expected exceptions
    with contextlib.suppress(KeyError):
        {}["missing"]
    print("KeyError suppressed via contextlib.suppress")


# ============================================================================
# 22. TEMPFILE QUICK HITS
# ============================================================================

def tempfile_quick_hits():
    """TemporaryDirectory and NamedTemporaryFile basics."""
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        file_path = tmp_path / "demo.txt"
        file_path.write_text("temp data", encoding="utf-8")
        print("Temp dir:", tmp_path)
        print("Temp file exists:", file_path.exists())

    # NamedTemporaryFile auto-deletes on close by default
    with tempfile.NamedTemporaryFile(delete=True) as f:
        f.write(b"bytes here")
        f.flush()
        print("NamedTemporaryFile path:", f.name)


# ============================================================================
# 23. PROFILING & DEBUGGING
# ============================================================================

def profiling_and_debugging():
    """timeit, cProfile, and breakpoint note."""
    import timeit, cProfile, pstats, io

    def work():
        return sum(i * i for i in range(10_000))

    timing = timeit.timeit(work, number=100)
    print(f"timeit 100 runs: {timing:.4f}s")

    profiler = cProfile.Profile()
    profiler.enable()
    work()
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
    ps.print_stats(5)
    print("cProfile sample:\n", s.getvalue())

    print("Use breakpoint() to drop into the debugger where needed.")


# ============================================================================
# 24. ASYNCIO ADVANCED PATTERNS
# ============================================================================

async def asyncio_advanced_patterns():
    """Semaphores, create_task, cancellation."""
    import asyncio

    sem = asyncio.Semaphore(2)

    async def limited(name, delay):
        async with sem:
            print(f"{name} start")
            await asyncio.sleep(delay)
            print(f"{name} end")
            return name

    # Demonstrate cancellation
    async def cancellable():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            print("cancellable task cancelled")
            raise

    tasks = [asyncio.create_task(limited(f"job-{i}", 0.3 + i * 0.1)) for i in range(4)]
    cancel_task = asyncio.create_task(cancellable())
    asyncio.get_running_loop().call_later(0.5, cancel_task.cancel)

    results = await asyncio.gather(*tasks, return_exceptions=False)
    try:
        await cancel_task
    except asyncio.CancelledError:
        pass

    print("Semaphore-limited results:", results)


def run_asyncio_advanced_patterns():
    import asyncio
    asyncio.run(asyncio_advanced_patterns())


# ============================================================================
# 25. TESTING EXTRAS (mock, parametrization mindset)
# ============================================================================

def testing_extras_mock_and_param():
    """unittest.mock basics and notes on parametrized tests."""
    from unittest import mock

    class Service:
        def fetch(self):
            return "real"

    svc = Service()
    with mock.patch.object(Service, "fetch", return_value="fake") as patched:
        assert svc.fetch() == "fake"
        print("mock called:", patched.called)

    print("Parametrization tip: in pytest use @pytest.mark.parametrize; in unittest loop over cases or subTest().")


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

    # 10. Files & Paths
    # files_and_paths()

    # 11. CSV & JSON
    # csv_and_json_examples()

    # 12. Logging Basics
    # logging_basics()

    # 13. Concurrency Quick Hits
    # threading_quick_demo()
    # threadpool_quick_demo()
    # multiprocessing_quick_demo()
    # run_asyncio_quick_demo()
    
    # 14. Testing
    # testing_unittest_basics()
    # testing_pytest_style_examples()
    
    # 15. Packaging & Environments
    # packaging_and_venv_notes()

    # 16. CLI (argparse) quick start
    # argparse_quick_start()

    # 17. Datetime & timezones
    # datetime_and_timezones()

    # 18. Regex essentials
    # regex_essentials()

    # 19. HTTP basics
    # http_basics()

    # 20. SQLite mini demo
    # sqlite_mini_demo()

    # 21. contextlib utilities
    # contextlib_utilities()

    # 22. tempfile quick hits
    # tempfile_quick_hits()

    # 23. Profiling & debugging
    # profiling_and_debugging()

    # 24. Asyncio advanced patterns
    # run_asyncio_advanced_patterns()

    # 25. Testing extras (mock, parametrization)
    # testing_extras_mock_and_param()

    print("\nParts 1-25 ready. Uncomment a section above to run its demo.")
    print("=" * 70)
