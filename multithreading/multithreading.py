"""
================================================================================
PYTHON MULTITHREADING & CONCURRENCY CHEAT SHEET
================================================================================

COMPREHENSIVE GUIDE covering threading, multiprocessing, async programming,
and all forms of concurrent execution in Python.

WHAT IS MULTITHREADING?
-----------------------
Multithreading allows a program to do multiple things at the same time.
Instead of running tasks one after another (sequentially), you can run them
concurrently (overlapping in time).

WHY USE MULTITHREADING?
-----------------------
1. Speed up I/O-bound tasks (file reading, network requests, database queries)
2. Keep UI responsive while doing background work
3. Handle multiple clients/requests simultaneously
4. Improve resource utilization

KEY CONCEPTS:
-------------
- Thread: Lightweight execution unit that shares memory with other threads
- Process: Heavy execution unit with its own memory space
- GIL (Global Interpreter Lock): Python limitation preventing true parallel
  execution of Python code in threads (but I/O operations release the GIL)
- Race Condition: When multiple threads access shared data simultaneously
- Deadlock: When threads wait for each other forever
- Lock/Mutex: Mechanism to ensure only one thread accesses data at a time

LIBRARIES COVERED:
------------------
1. threading - Standard library for thread-based concurrency
2. multiprocessing - Process-based parallelism (bypasses GIL)
3. concurrent.futures - High-level interface (ThreadPoolExecutor, ProcessPoolExecutor)
4. asyncio - Async/await for cooperative multitasking
5. queue - Thread-safe data structures
6. Lock, RLock, Semaphore, Event - Synchronization primitives

CONTENTS:
---------
1.  Threading Basics: Creating and running threads
2.  Thread Synchronization: Locks, RLocks, Semaphores
3.  Thread Communication: Queues, Events, Conditions
4.  ThreadPoolExecutor: Managing thread pools efficiently
5.  Multiprocessing: Process-based parallelism
6.  ProcessPoolExecutor: Managing process pools
7.  Async/Await (asyncio): Cooperative multitasking
8.  Common Patterns: Producer-Consumer, Worker Pools
9.  Thread Safety: Avoiding race conditions
10. Performance Comparison: When to use what
11. Best Practices: Tips and pitfalls to avoid
12. Real-World Examples: Web scraping, file processing, API calls

WHEN TO USE WHAT:
-----------------
- Threading: I/O-bound tasks (network, files, databases)
- Multiprocessing: CPU-bound tasks (calculations, data processing)
- Asyncio: Many I/O-bound tasks, web servers, network protocols
- Sequential: Simple tasks, small workloads

================================================================================
"""

# ============================================================================
# IMPORTS - Core Libraries for Concurrency
# ============================================================================

import threading           # Thread-based concurrency (shares memory)
import multiprocessing    # Process-based parallelism (separate memory)
import concurrent.futures # High-level interface for threads/processes
import asyncio            # Async/await for cooperative multitasking
import time               # Timing and sleep functions
import queue              # Thread-safe FIFO queue
import os                 # Operating system interface (process IDs)
from threading import Thread, Lock, RLock, Semaphore, Event, Condition
from queue import Queue
import requests           # HTTP library for examples (may need: pip install requests)
import random             # Random numbers for examples

# ============================================================================
# 1. THREADING BASICS
# ============================================================================
# Threads are lightweight and share memory. Good for I/O-bound tasks.

def basic_thread_example():
    """Create and run a simple thread"""
    
    def worker(name, duration):
        """Function that will run in a separate thread"""
        print(f"Thread {name}: Starting work")
        time.sleep(duration)  # Simulate work (I/O operation)
        print(f"Thread {name}: Finished after {duration} seconds")
    
    # Create thread object
    thread1 = threading.Thread(
        target=worker,           # Function to run
        args=("A", 2)           # Arguments to pass to function
    )
    
    thread2 = threading.Thread(target=worker, args=("B", 1))
    
    # Start threads (begins execution)
    print("Main: Starting threads")
    thread1.start()  # Doesn't block - continues immediately
    thread2.start()
    
    # Wait for threads to complete
    print("Main: Waiting for threads to finish")
    thread1.join()  # Blocks until thread1 finishes
    thread2.join()  # Blocks until thread2 finishes
    
    print("Main: All threads completed")

def thread_with_return_value():
    """Get return values from threads using a queue"""
    
    def worker(n, result_queue):
        """Calculate and put result in queue"""
        result = n * n
        time.sleep(1)
        result_queue.put(result)  # Put result in thread-safe queue
    
    # Create queue to collect results
    results = queue.Queue()
    
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i, results))
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # Collect results
    print("Results:", [results.get() for _ in range(5)])

def daemon_threads():
    """Daemon threads run in background and die when main program exits"""
    
    def background_task():
        """Runs forever in background"""
        while True:
            print("Background: Still running...")
            time.sleep(1)
    
    # Create daemon thread
    daemon = threading.Thread(target=background_task)
    daemon.daemon = True  # Mark as daemon (won't prevent program exit)
    daemon.start()
    
    # Main program does some work
    time.sleep(3)
    print("Main: Exiting (daemon thread will be killed)")
    # When main exits, daemon thread stops automatically

# ============================================================================
# 2. THREAD SYNCHRONIZATION
# ============================================================================
# Prevent race conditions when multiple threads access shared data

def race_condition_example():
    """Demonstrates what happens WITHOUT proper synchronization"""
    
    counter = 0  # Shared variable
    
    def increment():
        """Increment counter 100,000 times"""
        nonlocal counter
        for _ in range(100000):
            # This is NOT atomic! Multiple operations:
            # 1. Read counter
            # 2. Add 1
            # 3. Write back
            counter += 1
    
    # Create two threads both incrementing same counter
    t1 = threading.Thread(target=increment)
    t2 = threading.Thread(target=increment)
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # Expected: 200,000 but actual will be less due to race condition!
    print(f"Counter without lock: {counter} (should be 200000)")

def lock_example():
    """Using Lock to prevent race conditions"""
    
    counter = 0
    lock = threading.Lock()  # Create lock object
    
    def increment():
        """Safely increment counter using lock"""
        nonlocal counter
        for _ in range(100000):
            with lock:  # Acquire lock (only one thread at a time)
                counter += 1
            # Lock released automatically when exiting 'with' block
    
    t1 = threading.Thread(target=increment)
    t2 = threading.Thread(target=increment)
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    print(f"Counter with lock: {counter} (should be 200000)")

def rlock_example():
    """RLock (Reentrant Lock) - same thread can acquire multiple times"""
    
    rlock = threading.RLock()  # Reentrant lock
    
    def outer():
        """Outer function acquires lock"""
        with rlock:
            print("Outer: Acquired lock")
            inner()  # Can call inner which also needs same lock!
    
    def inner():
        """Inner function also needs lock"""
        with rlock:  # Same thread can acquire again (reentrant)
            print("Inner: Acquired lock (same thread)")
    
    outer()  # Would deadlock with regular Lock, but works with RLock

def semaphore_example():
    """Semaphore limits number of threads accessing resource"""
    
    # Allow max 3 threads to access resource simultaneously
    semaphore = threading.Semaphore(3)
    
    def access_resource(worker_id):
        """Try to access limited resource"""
        print(f"Worker {worker_id}: Waiting for resource")
        
        with semaphore:  # Acquire semaphore (blocks if 3 already in use)
            print(f"Worker {worker_id}: Got resource!")
            time.sleep(2)  # Use resource
            print(f"Worker {worker_id}: Releasing resource")
        # Semaphore released, another thread can enter
    
    # Start 10 threads, but only 3 can run at once
    threads = []
    for i in range(10):
        t = threading.Thread(target=access_resource, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

# ============================================================================
# 3. THREAD COMMUNICATION
# ============================================================================
# Threads need to communicate and coordinate

def queue_example():
    """Queue for thread-safe communication"""
    
    task_queue = queue.Queue()  # Thread-safe FIFO queue
    
    def producer():
        """Produces items and puts them in queue"""
        for i in range(5):
            item = f"Item-{i}"
            print(f"Producer: Creating {item}")
            task_queue.put(item)  # Add to queue (thread-safe)
            time.sleep(0.5)
        task_queue.put(None)  # Sentinel value to signal completion
    
    def consumer():
        """Consumes items from queue"""
        while True:
            item = task_queue.get()  # Blocks until item available
            if item is None:  # Stop signal
                break
            print(f"Consumer: Processing {item}")
            time.sleep(1)
            task_queue.task_done()  # Mark task as complete
    
    # Start producer and consumer threads
    prod = threading.Thread(target=producer)
    cons = threading.Thread(target=consumer)
    
    prod.start()
    cons.start()
    
    prod.join()
    cons.join()

def event_example():
    """Event for signaling between threads"""
    
    event = threading.Event()  # Initially False (not set)
    
    def waiter():
        """Wait for event to be set"""
        print("Waiter: Waiting for event...")
        event.wait()  # Blocks until event is set
        print("Waiter: Event received! Proceeding...")
    
    def setter():
        """Set event after some time"""
        time.sleep(2)
        print("Setter: Setting event")
        event.set()  # Wake up all waiting threads
    
    t1 = threading.Thread(target=waiter)
    t2 = threading.Thread(target=setter)
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()

def condition_example():
    """Condition for complex thread coordination"""
    
    items = []
    condition = threading.Condition()  # Combines lock + event
    
    def consumer():
        """Wait for items to be available"""
        with condition:  # Acquire lock
            while len(items) == 0:  # Check condition
                print("Consumer: Waiting for items...")
                condition.wait()  # Release lock and wait for notification
                # Lock reacquired when notified
            
            item = items.pop(0)
            print(f"Consumer: Got {item}")
    
    def producer():
        """Produce item and notify consumer"""
        time.sleep(1)
        with condition:  # Acquire lock
            item = "Product"
            items.append(item)
            print(f"Producer: Created {item}")
            condition.notify()  # Wake up one waiting thread
    
    t1 = threading.Thread(target=consumer)
    t2 = threading.Thread(target=producer)
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()

# ============================================================================
# 4. THREADPOOLEXECUTOR
# ============================================================================
# High-level interface for managing pools of threads

def threadpool_basic():
    """ThreadPoolExecutor manages thread pool automatically"""
    from concurrent.futures import ThreadPoolExecutor
    
    def task(n):
        """Simple task that returns square"""
        print(f"Processing {n}")
        time.sleep(1)
        return n * n
    
    # Create pool with 3 worker threads
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks to pool
        futures = [executor.submit(task, i) for i in range(5)]
        
        # Get results as they complete
        for future in futures:
            result = future.result()  # Blocks until result available
            print(f"Result: {result}")

def threadpool_map():
    """Using map() for parallel processing"""
    from concurrent.futures import ThreadPoolExecutor
    
    def process_url(url):
        """Simulate downloading URL"""
        print(f"Downloading {url}")
        time.sleep(1)
        return f"Content from {url}"
    
    urls = [f"http://example.com/page{i}" for i in range(5)]
    
    # map() applies function to all items in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(process_url, urls)
        
        # Results maintain order of input
        for url, content in zip(urls, results):
            print(f"{url} -> {content}")

def threadpool_as_completed():
    """Process results as they complete (not in order)"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def task(n):
        """Task with variable duration"""
        duration = random.uniform(0.5, 2)
        time.sleep(duration)
        return f"Task {n} finished after {duration:.2f}s"
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        futures = {executor.submit(task, i): i for i in range(5)}
        
        # Process results as they complete (fastest first)
        for future in as_completed(futures):
            result = future.result()
            print(result)

# ============================================================================
# 5. MULTIPROCESSING
# ============================================================================
# Process-based parallelism - bypasses GIL, true parallel execution

def multiprocessing_basic():
    """Create and run processes (similar to threads but separate memory)"""
    
    def worker(name, duration):
        """Function runs in separate process"""
        pid = os.getpid()  # Each process has unique ID
        print(f"Process {name} (PID {pid}): Starting")
        time.sleep(duration)
        print(f"Process {name} (PID {pid}): Finished")
    
    # Create processes (not threads!)
    p1 = multiprocessing.Process(target=worker, args=("A", 2))
    p2 = multiprocessing.Process(target=worker, args=("B", 1))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
    print("Main: All processes completed")

def multiprocessing_with_queue():
    """Communication between processes using Queue"""
    
    def worker(task_queue, result_queue):
        """Process tasks from queue"""
        while True:
            task = task_queue.get()
            if task is None:  # Stop signal
                break
            
            # Do computation (this runs in parallel!)
            result = task * task
            result_queue.put(result)
    
    # Multiprocessing Queue (different from threading.Queue)
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    
    # Start worker processes
    processes = []
    for _ in range(3):
        p = multiprocessing.Process(target=worker, args=(tasks, results))
        p.start()
        processes.append(p)
    
    # Add tasks
    for i in range(10):
        tasks.put(i)
    
    # Send stop signals
    for _ in range(3):
        tasks.put(None)
    
    # Wait for completion
    for p in processes:
        p.join()
    
    # Collect results
    output = []
    while not results.empty():
        output.append(results.get())
    
    print(f"Results: {sorted(output)}")

def multiprocessing_pool():
    """Pool of worker processes"""
    
    def cpu_intensive_task(n):
        """Simulate CPU-intensive work"""
        result = sum(i * i for i in range(n))
        return result
    
    # Create pool of worker processes
    with multiprocessing.Pool(processes=4) as pool:
        # Map function across inputs in parallel
        inputs = [1000000, 2000000, 3000000, 4000000]
        results = pool.map(cpu_intensive_task, inputs)
        
        print(f"Results: {results}")

# ============================================================================
# 6. PROCESSPOOLEXECUTOR
# ============================================================================
# High-level interface for process pools (similar to ThreadPoolExecutor)

def processpoolexecutor_example():
    """ProcessPoolExecutor for CPU-bound tasks"""
    from concurrent.futures import ProcessPoolExecutor
    
    def fibonacci(n):
        """Calculate nth Fibonacci number (CPU-intensive)"""
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)
    
    # Use processes for CPU-bound work
    with ProcessPoolExecutor(max_workers=4) as executor:
        numbers = [30, 31, 32, 33, 34]
        
        # Calculate in parallel across CPU cores
        results = executor.map(fibonacci, numbers)
        
        for num, result in zip(numbers, results):
            print(f"Fibonacci({num}) = {result}")

# ============================================================================
# 7. ASYNC/AWAIT (ASYNCIO)
# ============================================================================
# Cooperative multitasking - single thread, but can handle many I/O tasks

async def async_basic():
    """Basic async/await example"""
    
    async def task(name, duration):
        """Async function (coroutine)"""
        print(f"Task {name}: Starting")
        await asyncio.sleep(duration)  # Yields control to event loop
        print(f"Task {name}: Finished")
        return f"Result from {name}"
    
    # Run multiple coroutines concurrently
    results = await asyncio.gather(
        task("A", 2),
        task("B", 1),
        task("C", 1.5)
    )
    
    print(f"All tasks completed: {results}")

async def async_with_requests():
    """Async HTTP requests (requires aiohttp: pip install aiohttp)"""
    # Note: This is example code structure
    # import aiohttp
    
    async def fetch_url(session, url):
        """Fetch URL asynchronously"""
        # async with session.get(url) as response:
        #     return await response.text()
        
        # Simulated version:
        await asyncio.sleep(1)
        return f"Content from {url}"
    
    # async with aiohttp.ClientSession() as session:
    #     urls = ["http://example.com/1", "http://example.com/2"]
    #     tasks = [fetch_url(session, url) for url in urls]
    #     results = await asyncio.gather(*tasks)
    
    print("Async requests would run concurrently here")

def run_async_example():
    """Helper to run async code from synchronous code"""
    # Run async function
    asyncio.run(async_basic())

# ============================================================================
# 8. COMMON PATTERNS
# ============================================================================
# Practical patterns you'll use frequently

def producer_consumer_pattern():
    """Classic producer-consumer with thread pool"""
    
    def producer(queue, n_items):
        """Produce items"""
        for i in range(n_items):
            item = f"Item-{i}"
            queue.put(item)
            print(f"Produced: {item}")
            time.sleep(0.1)
        
        # Poison pill to signal completion
        queue.put(None)
    
    def consumer(queue, worker_id):
        """Consume items until poison pill"""
        while True:
            item = queue.get()
            if item is None:
                queue.put(None)  # Pass poison pill to next consumer
                break
            
            print(f"Consumer {worker_id}: Processing {item}")
            time.sleep(0.5)
            queue.task_done()
    
    q = queue.Queue()
    
    # One producer
    prod = threading.Thread(target=producer, args=(q, 10))
    
    # Multiple consumers
    consumers = [
        threading.Thread(target=consumer, args=(q, i))
        for i in range(3)
    ]
    
    prod.start()
    for c in consumers:
        c.start()
    
    prod.join()
    for c in consumers:
        c.join()

def worker_pool_pattern():
    """Worker pool for batch processing"""
    from concurrent.futures import ThreadPoolExecutor
    
    def process_file(filename):
        """Simulate file processing"""
        print(f"Processing {filename}")
        time.sleep(1)
        return f"Processed {filename}"
    
    files = [f"file{i}.txt" for i in range(10)]
    
    # Process files in parallel with worker pool
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, files))
    
    print(f"Completed: {len(results)} files")

# ============================================================================
# 9. THREAD SAFETY
# ============================================================================
# Writing code that works correctly with multiple threads

def thread_local_storage():
    """Each thread has its own copy of data"""
    
    # Thread-local storage
    thread_local = threading.local()
    
    def worker(value):
        """Each thread stores its own value"""
        thread_local.data = value  # Separate for each thread
        time.sleep(1)
        print(f"Thread data: {thread_local.data}")
    
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

def atomic_operations():
    """Some operations are atomic in Python"""
    
    # Atomic (thread-safe without lock):
    # - Reading/writing simple types (int, str, etc.)
    # - list.append()
    # - dict.__setitem__()
    
    # NOT atomic (need lock):
    # - counter += 1 (read-modify-write)
    # - list.extend()
    # - Most operations on mutable objects
    
    shared_list = []  # Thread-safe for append
    
    def append_items():
        for i in range(1000):
            shared_list.append(i)  # Atomic operation
    
    threads = [threading.Thread(target=append_items) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    print(f"List length: {len(shared_list)}")  # Should be 3000

# ============================================================================
# 10. PERFORMANCE COMPARISON
# ============================================================================
# When to use threading vs multiprocessing vs asyncio

def compare_approaches():
    """Compare different concurrency approaches"""
    
    def io_task():
        """I/O-bound task (network, file, database)"""
        time.sleep(0.1)  # Simulates I/O wait
        return "done"
    
    def cpu_task():
        """CPU-bound task (calculations)"""
        return sum(i * i for i in range(100000))
    
    # Sequential
    start = time.time()
    for _ in range(10):
        io_task()
    print(f"Sequential I/O: {time.time() - start:.2f}s")
    
    # Threading (good for I/O)
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(lambda x: io_task(), range(10)))
    print(f"Threading I/O: {time.time() - start:.2f}s")
    
    # Multiprocessing (good for CPU)
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda x: cpu_task(), range(10)))
    print(f"Multiprocessing CPU: {time.time() - start:.2f}s")

# ============================================================================
# 11. BEST PRACTICES
# ============================================================================

"""
THREADING BEST PRACTICES:
-------------------------
1. Use ThreadPoolExecutor instead of manual thread management
2. Always use locks for shared mutable data
3. Prefer Queue for thread communication
4. Use daemon threads for background tasks
5. Avoid global state when possible
6. Use context managers (with) for locks
7. Don't create too many threads (use pool)
8. Join threads to prevent resource leaks

MULTIPROCESSING BEST PRACTICES:
-------------------------------
1. Use for CPU-intensive tasks only
2. Minimize data passed between processes
3. Use Pool for batch processing
4. Protect main code with if __name__ == '__main__'
5. Use Manager() for shared state (slow)
6. Processes have overhead - use when benefit > cost

ASYNCIO BEST PRACTICES:
----------------------
1. Use for I/O-bound tasks with many connections
2. Don't block the event loop (use async libraries)
3. Use asyncio.gather() for concurrent tasks
4. Handle exceptions properly
5. Use async context managers (async with)

AVOID THESE PITFALLS:
--------------------
1. Race conditions (forgetting locks)
2. Deadlocks (circular lock dependencies)
3. Using threading for CPU-bound tasks
4. Sharing non-thread-safe objects
5. Not joining threads (resource leak)
6. Creating too many threads/processes
7. Ignoring the GIL in threading
"""

# ============================================================================
# 12. REAL-WORLD EXAMPLES
# ============================================================================

def web_scraper_example():
    """Scrape multiple URLs concurrently"""
    from concurrent.futures import ThreadPoolExecutor
    
    def fetch_url(url):
        """Fetch and process URL"""
        try:
            # In real code: response = requests.get(url, timeout=5)
            # Simulated:
            time.sleep(1)
            return f"Downloaded {url}"
        except Exception as e:
            return f"Error: {e}"
    
    urls = [
        "http://example.com/page1",
        "http://example.com/page2",
        "http://example.com/page3",
        "http://example.com/page4",
    ]
    
    # Use thread pool for I/O-bound web scraping
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(fetch_url, urls)
        
        for url, result in zip(urls, results):
            print(f"{url}: {result}")

def file_processor_example():
    """Process multiple files in parallel"""
    from concurrent.futures import ProcessPoolExecutor
    
    def process_file(filepath):
        """CPU-intensive file processing"""
        # Simulated processing
        data = sum(i * i for i in range(1000000))
        return f"{filepath}: {data}"
    
    files = [f"data_{i}.csv" for i in range(5)]
    
    # Use processes for CPU-intensive work
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_file, files)
        
        for result in results:
            print(result)

def api_client_example():
    """Make multiple API calls concurrently"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def call_api(endpoint):
        """Call API endpoint"""
        # Simulated API call
        time.sleep(random.uniform(0.5, 2))
        return {"endpoint": endpoint, "data": f"Response from {endpoint}"}
    
    endpoints = [f"/api/users/{i}" for i in range(10)]
    
    # Use threads for I/O-bound API calls
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all requests
        futures = {executor.submit(call_api, ep): ep for ep in endpoints}
        
        # Process results as they complete
        for future in as_completed(futures):
            endpoint = futures[future]
            try:
                result = future.result()
                print(f"Success: {result}")
            except Exception as e:
                print(f"Error for {endpoint}: {e}")

def download_manager_example():
    """Download manager with progress tracking"""
    
    def download_file(url, progress_queue):
        """Download file with progress updates"""
        filename = url.split('/')[-1]
        
        # Simulate download
        for i in range(10):
            time.sleep(0.2)
            progress = (i + 1) * 10
            progress_queue.put((filename, progress))
        
        return filename
    
    def progress_monitor(progress_queue, n_files):
        """Monitor download progress"""
        completed = 0
        while completed < n_files:
            filename, progress = progress_queue.get()
            print(f"{filename}: {progress}% complete")
            if progress == 100:
                completed += 1
    
    urls = [
        "http://example.com/file1.zip",
        "http://example.com/file2.zip",
        "http://example.com/file3.zip"
    ]
    
    progress_q = queue.Queue()
    
    # Start progress monitor
    monitor = threading.Thread(
        target=progress_monitor,
        args=(progress_q, len(urls))
    )
    monitor.start()
    
    # Download files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(download_file, url, progress_q)
            for url in urls
        ]
        
        # Wait for downloads
        for future in concurrent.futures.as_completed(futures):
            filename = future.result()
            print(f"Download complete: {filename}")
    
    monitor.join()

# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PYTHON MULTITHREADING & CONCURRENCY CHEAT SHEET")
    print("=" * 70)
    print("\nThis file covers everything you need to know about:")
    print("  - Threading (concurrent I/O)")
    print("  - Multiprocessing (parallel CPU work)")
    print("  - Async/await (cooperative multitasking)")
    print("  - Thread safety and synchronization")
    print("  - Real-world patterns and examples")
    print("\nUncomment function calls below to run examples.")
    print("=" * 70)
    
    # Section 1: Threading Basics
    print("\n### 1. THREADING BASICS ###")
    # basic_thread_example()
    # thread_with_return_value()
    # daemon_threads()
    
    # Section 2: Synchronization
    print("\n### 2. THREAD SYNCHRONIZATION ###")
    # race_condition_example()
    # lock_example()
    # rlock_example()
    # semaphore_example()
    
    # Section 3: Communication
    print("\n### 3. THREAD COMMUNICATION ###")
    # queue_example()
    # event_example()
    # condition_example()
    
    # Section 4: ThreadPoolExecutor
    print("\n### 4. THREADPOOLEXECUTOR ###")
    # threadpool_basic()
    # threadpool_map()
    # threadpool_as_completed()
    
    # Section 5: Multiprocessing
    print("\n### 5. MULTIPROCESSING ###")
    # multiprocessing_basic()
    # multiprocessing_with_queue()
    # multiprocessing_pool()
    
    # Section 6: ProcessPoolExecutor
    print("\n### 6. PROCESSPOOLEXECUTOR ###")
    # processpoolexecutor_example()
    
    # Section 7: Async/Await
    print("\n### 7. ASYNC/AWAIT ###")
    # run_async_example()
    
    # Section 8: Common Patterns
    print("\n### 8. COMMON PATTERNS ###")
    # producer_consumer_pattern()
    # worker_pool_pattern()
    
    # Section 9: Thread Safety
    print("\n### 9. THREAD SAFETY ###")
    # thread_local_storage()
    # atomic_operations()
    
    # Section 10: Performance
    print("\n### 10. PERFORMANCE COMPARISON ###")
    # compare_approaches()
    
    # Section 12: Real-World Examples
    print("\n### 12. REAL-WORLD EXAMPLES ###")
    # web_scraper_example()
    # file_processor_example()
    # api_client_example()
    # download_manager_example()
    
    print("\n" + "=" * 70)
    print("All examples loaded successfully!")
    print("Uncomment function calls to run them.")
    print("=" * 70)
