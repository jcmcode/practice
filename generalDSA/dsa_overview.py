"""
DATA STRUCTURES & ALGORITHMS - COMPREHENSIVE CHEAT SHEET
=========================================================
Part 1: Core Data Structures
This file covers fundamental data structures and their operations in Python.
"""

from collections import deque, defaultdict, Counter, OrderedDict
from typing import List, Optional, Any
import heapq

# ============================================================================
# 1. ARRAYS / LISTS
# ============================================================================
"""
Arrays (Lists in Python):
- Dynamic array implementation
- Contiguous memory allocation
- O(1) access by index
- O(1) append (amortized)
- O(n) insert/delete at arbitrary position
"""

def array_operations():
    """Comprehensive list operations and techniques"""
    
    # --- Creation ---
    arr = [1, 2, 3, 4, 5]
    arr_zeros = [0] * 5  # [0, 0, 0, 0, 0]
    arr_range = list(range(5))  # [0, 1, 2, 3, 4]
    arr_2d = [[0] * 3 for _ in range(2)]  # [[0,0,0], [0,0,0]]
    
    # IMPORTANT: Wrong way for 2D arrays (creates references to same list)
    # wrong_2d = [[0] * 3] * 2  # DON'T DO THIS!
    
    # --- Access ---
    first = arr[0]          # First element: O(1)
    last = arr[-1]          # Last element: O(1)
    second_last = arr[-2]   # Second to last: O(1)
    
    # --- Slicing ---
    # arr[start:end:step] - end is exclusive
    subset = arr[1:4]       # [2, 3, 4]
    first_three = arr[:3]   # [1, 2, 3]
    last_three = arr[-3:]   # [3, 4, 5]
    reverse = arr[::-1]     # [5, 4, 3, 2, 1]
    every_second = arr[::2] # [1, 3, 5]
    copy = arr[:]           # Shallow copy
    
    # --- Modification ---
    arr.append(6)           # Add to end: O(1)
    arr.insert(0, 0)        # Insert at position: O(n)
    arr.extend([7, 8])      # Add multiple: O(k) where k is length
    arr += [9, 10]          # Same as extend
    
    # --- Removal ---
    arr.pop()               # Remove and return last: O(1)
    arr.pop(0)              # Remove at index: O(n)
    arr.remove(5)           # Remove first occurrence: O(n)
    del arr[0]              # Delete at index: O(n)
    arr.clear()             # Remove all: O(n)
    
    # --- Searching ---
    arr = [1, 2, 3, 2, 4]
    idx = arr.index(2)      # First index of 2: O(n)
    count = arr.count(2)    # Count occurrences: O(n)
    exists = 3 in arr       # Check if exists: O(n)
    
    # --- Sorting ---
    arr.sort()              # In-place sort: O(n log n)
    arr.sort(reverse=True)  # Descending
    sorted_arr = sorted(arr)  # Returns new sorted list
    
    # Custom sorting
    words = ["apple", "pie", "zoo", "a"]
    words.sort(key=len)     # Sort by length
    words.sort(key=lambda x: x[0])  # Sort by first character
    
    # --- Other Operations ---
    arr.reverse()           # In-place reverse: O(n)
    length = len(arr)       # Length: O(1)
    max_val = max(arr)      # Maximum: O(n)
    min_val = min(arr)      # Minimum: O(n)
    total = sum(arr)        # Sum: O(n)
    
    # --- List Comprehension ---
    squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]
    evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
    matrix = [[i+j for j in range(3)] for i in range(3)]
    
    # --- Common Patterns ---
    # Two pointers
    left, right = 0, len(arr) - 1
    
    # Sliding window
    window_sum = sum(arr[:3])  # Initial window
    
    # Enumerate (get index and value)
    for i, val in enumerate(arr):
        print(f"Index {i}: {val}")
    
    # Zip (combine multiple lists)
    list1 = [1, 2, 3]
    list2 = ['a', 'b', 'c']
    combined = list(zip(list1, list2))  # [(1,'a'), (2,'b'), (3,'c')]
    
    # All/Any
    all_positive = all(x > 0 for x in arr)
    any_even = any(x % 2 == 0 for x in arr)
    
    return arr

# ============================================================================
# 2. STRINGS
# ============================================================================
"""
Strings:
- Immutable sequence of characters
- O(1) access by index
- O(n) for most operations due to immutability
"""

def string_operations():
    """Comprehensive string operations"""
    
    s = "Hello World"
    
    # --- Access ---
    first_char = s[0]       # 'H'
    last_char = s[-1]       # 'd'
    substring = s[0:5]      # 'Hello'
    
    # --- Immutability ---
    # s[0] = 'h'  # ERROR! Strings are immutable
    # Must create new string
    s_lower = 'h' + s[1:]   # 'hello World'
    
    # --- Common Methods ---
    upper = s.upper()       # 'HELLO WORLD'
    lower = s.lower()       # 'hello world'
    title = s.title()       # 'Hello World'
    swapped = s.swapcase()  # 'hELLO wORLD'
    
    # --- Strip/Trim ---
    s2 = "  hello  "
    stripped = s2.strip()   # 'hello'
    lstrip = s2.lstrip()    # 'hello  '
    rstrip = s2.rstrip()    # '  hello'
    
    # --- Split/Join ---
    words = s.split()       # ['Hello', 'World']
    words2 = s.split('o')   # ['Hell', ' W', 'rld']
    joined = '-'.join(words)  # 'Hello-World'
    
    # --- Find/Search ---
    idx = s.find('World')   # 6 (returns -1 if not found)
    idx2 = s.index('World') # 6 (raises ValueError if not found)
    count = s.count('l')    # 3
    starts = s.startswith('Hello')  # True
    ends = s.endswith('World')      # True
    
    # --- Replace ---
    replaced = s.replace('World', 'Python')  # 'Hello Python'
    
    # --- Check Properties ---
    is_alpha = s.isalpha()      # False (has space)
    is_digit = '123'.isdigit()  # True
    is_alnum = 'abc123'.isalnum()  # True
    is_space = '   '.isspace()  # True
    is_upper = 'ABC'.isupper()  # True
    is_lower = 'abc'.islower()  # True
    
    # --- String Building (efficient way) ---
    # DON'T: result = ""
    #        for char in s: result += char  # O(n²) - creates new string each time
    
    # DO: Use list and join
    chars = []
    for char in s:
        chars.append(char)
    result = ''.join(chars)  # O(n)
    
    # --- Common Patterns ---
    # Reverse string
    reversed_s = s[::-1]
    
    # Character array conversion
    char_list = list(s)
    
    # Check palindrome
    is_palindrome = s == s[::-1]
    
    # Count character frequency
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    
    # Or use Counter
    from collections import Counter
    freq2 = Counter(s)
    
    # String formatting
    name, age = "Alice", 25
    formatted1 = f"{name} is {age} years old"  # f-strings (Python 3.6+)
    formatted2 = "{} is {} years old".format(name, age)
    formatted3 = "%s is %d years old" % (name, age)
    
    return result

# ============================================================================
# 3. DICTIONARIES / HASH MAPS
# ============================================================================
"""
Dictionaries (Hash Maps):
- Key-value pairs
- O(1) average case for insert, delete, lookup
- O(n) worst case (rare with good hash function)
- Keys must be immutable (hashable)
- Maintains insertion order (Python 3.7+)
"""

def dictionary_operations():
    """Comprehensive dictionary operations"""
    
    # --- Creation ---
    d = {'a': 1, 'b': 2, 'c': 3}
    d2 = dict(a=1, b=2, c=3)
    d3 = dict([('a', 1), ('b', 2)])
    d4 = {x: x**2 for x in range(5)}  # Dict comprehension
    
    # --- Access ---
    val = d['a']            # 1 (KeyError if not exists)
    val2 = d.get('a')       # 1
    val3 = d.get('z', 0)    # 0 (default value)
    
    # --- Modification ---
    d['d'] = 4              # Add/update: O(1)
    d.update({'e': 5, 'f': 6})  # Update multiple
    
    # --- Removal ---
    del d['a']              # Delete key: O(1)
    val = d.pop('b')        # Remove and return value
    val2 = d.pop('z', None) # With default if not exists
    d.popitem()             # Remove and return last inserted pair
    d.clear()               # Remove all
    
    # --- Checking ---
    exists = 'a' in d       # Check key exists: O(1)
    not_exists = 'z' not in d
    
    # --- Iteration ---
    d = {'a': 1, 'b': 2, 'c': 3}
    
    for key in d:           # Iterate keys
        print(key)
    
    for key in d.keys():    # Same as above
        print(key)
    
    for val in d.values():  # Iterate values
        print(val)
    
    for key, val in d.items():  # Iterate key-value pairs
        print(f"{key}: {val}")
    
    # --- Views ---
    keys = list(d.keys())       # ['a', 'b', 'c']
    values = list(d.values())   # [1, 2, 3]
    items = list(d.items())     # [('a',1), ('b',2), ('c',3)]
    
    # --- Common Patterns ---
    # Default value initialization
    d = {}
    for char in "hello":
        if char not in d:
            d[char] = 0
        d[char] += 1
    
    # Better: use get
    d = {}
    for char in "hello":
        d[char] = d.get(char, 0) + 1
    
    # Best: use defaultdict
    from collections import defaultdict
    d = defaultdict(int)
    for char in "hello":
        d[char] += 1
    
    # Invert dictionary (swap keys and values)
    original = {'a': 1, 'b': 2}
    inverted = {v: k for k, v in original.items()}
    
    # Merge dictionaries (Python 3.9+)
    d1 = {'a': 1, 'b': 2}
    d2 = {'c': 3, 'd': 4}
    merged = d1 | d2
    
    # Merge (Python 3.5+)
    merged2 = {**d1, **d2}
    
    # Sort dictionary by key
    sorted_dict = dict(sorted(d.items()))
    
    # Sort by value
    sorted_by_val = dict(sorted(d.items(), key=lambda x: x[1]))
    
    return d

# ============================================================================
# 4. SETS
# ============================================================================
"""
Sets:
- Unordered collection of unique elements
- O(1) average for add, remove, lookup
- Useful for membership testing, removing duplicates
- Elements must be immutable (hashable)
"""

def set_operations():
    """Comprehensive set operations"""
    
    # --- Creation ---
    s = {1, 2, 3, 4, 5}
    s2 = set([1, 2, 2, 3, 3])  # {1, 2, 3} - duplicates removed
    s3 = {x for x in range(5)}  # Set comprehension
    empty = set()  # Note: {} creates empty dict, not set!
    
    # --- Add/Remove ---
    s.add(6)                # Add element: O(1)
    s.update([7, 8, 9])     # Add multiple
    s.remove(5)             # Remove (KeyError if not exists)
    s.discard(5)            # Remove (no error if not exists)
    elem = s.pop()          # Remove and return arbitrary element
    s.clear()               # Remove all
    
    # --- Membership ---
    exists = 3 in s         # Check if exists: O(1)
    
    # --- Set Operations ---
    s1 = {1, 2, 3, 4, 5}
    s2 = {4, 5, 6, 7, 8}
    
    # Union: Elements in either set
    union = s1 | s2         # {1, 2, 3, 4, 5, 6, 7, 8}
    union2 = s1.union(s2)
    
    # Intersection: Elements in both sets
    inter = s1 & s2         # {4, 5}
    inter2 = s1.intersection(s2)
    
    # Difference: Elements in s1 but not s2
    diff = s1 - s2          # {1, 2, 3}
    diff2 = s1.difference(s2)
    
    # Symmetric Difference: Elements in either but not both
    sym_diff = s1 ^ s2      # {1, 2, 3, 6, 7, 8}
    sym_diff2 = s1.symmetric_difference(s2)
    
    # --- Subset/Superset ---
    s1 = {1, 2, 3}
    s2 = {1, 2, 3, 4, 5}
    is_subset = s1 <= s2    # True (s1 is subset of s2)
    is_proper_subset = s1 < s2  # True
    is_superset = s2 >= s1  # True (s2 is superset of s1)
    is_proper_superset = s2 > s1  # True
    
    # --- Common Patterns ---
    # Remove duplicates from list
    lst = [1, 2, 2, 3, 3, 4]
    unique = list(set(lst))
    
    # Find common elements
    list1 = [1, 2, 3, 4]
    list2 = [3, 4, 5, 6]
    common = set(list1) & set(list2)
    
    # Find unique elements in each
    only_in_1 = set(list1) - set(list2)
    only_in_2 = set(list2) - set(list1)
    
    return s

# ============================================================================
# 5. TUPLES
# ============================================================================
"""
Tuples:
- Immutable sequences
- O(1) access by index
- Hashable (can be dict keys or set elements)
- More memory efficient than lists
"""

def tuple_operations():
    """Comprehensive tuple operations"""
    
    # --- Creation ---
    t = (1, 2, 3, 4, 5)
    t2 = 1, 2, 3  # Parentheses optional
    single = (1,)  # Note the comma! (1) is just int 1
    empty = ()
    from_list = tuple([1, 2, 3])
    
    # --- Access ---
    first = t[0]
    last = t[-1]
    subset = t[1:4]  # (2, 3, 4)
    
    # --- Immutability ---
    # t[0] = 10  # ERROR! Tuples are immutable
    # Must create new tuple
    t_new = (10,) + t[1:]
    
    # --- Operations ---
    count = t.count(2)      # Count occurrences
    idx = t.index(3)        # Find index
    length = len(t)
    
    # --- Concatenation ---
    t1 = (1, 2)
    t2 = (3, 4)
    combined = t1 + t2      # (1, 2, 3, 4)
    repeated = t1 * 3       # (1, 2, 1, 2, 1, 2)
    
    # --- Unpacking ---
    a, b, c = (1, 2, 3)     # a=1, b=2, c=3
    first, *rest = (1, 2, 3, 4)  # first=1, rest=[2,3,4]
    first, *middle, last = (1, 2, 3, 4, 5)  # first=1, middle=[2,3,4], last=5
    
    # --- Common Uses ---
    # Return multiple values from function
    def get_coords():
        return (10, 20)
    
    x, y = get_coords()
    
    # As dictionary keys
    locations = {(0, 0): 'origin', (1, 1): 'point1'}
    
    # Named tuples (more readable)
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(10, 20)
    print(p.x, p.y)  # Access by name
    print(p[0], p[1])  # Can still access by index
    
    return t

# ============================================================================
# 6. STACKS
# ============================================================================
"""
Stack:
- LIFO (Last In First Out)
- O(1) push, pop, peek
- Use list or deque
- Applications: Function calls, undo/redo, expression evaluation, DFS
"""

class Stack:
    """Stack implementation using list"""
    
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top of stack - O(1)"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item - O(1)"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        """Return top item without removing - O(1)"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]
    
    def is_empty(self):
        """Check if stack is empty - O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items - O(1)"""
        return len(self.items)
    
    def __str__(self):
        return str(self.items)

def stack_examples():
    """Common stack problems and patterns"""
    
    # --- Using list as stack ---
    stack = []
    stack.append(1)  # push
    stack.append(2)
    stack.append(3)
    top = stack[-1]  # peek
    val = stack.pop()  # pop
    
    # --- Balanced Parentheses ---
    def is_balanced(s):
        """Check if parentheses are balanced"""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in s:
            if char in pairs:  # Opening bracket
                stack.append(char)
            elif char in pairs.values():  # Closing bracket
                if not stack or pairs[stack.pop()] != char:
                    return False
        
        return len(stack) == 0
    
    print(is_balanced("()[]{}"))  # True
    print(is_balanced("([)]"))    # False
    
    # --- Evaluate Postfix Expression ---
    def eval_postfix(expression):
        """Evaluate postfix expression like '2 3 + 4 *'"""
        stack = []
        operators = {'+', '-', '*', '/'}
        
        for token in expression.split():
            if token not in operators:
                stack.append(int(token))
            else:
                b = stack.pop()
                a = stack.pop()
                if token == '+':
                    stack.append(a + b)
                elif token == '-':
                    stack.append(a - b)
                elif token == '*':
                    stack.append(a * b)
                elif token == '/':
                    stack.append(a // b)
        
        return stack[0]
    
    print(eval_postfix("2 3 + 4 *"))  # 20
    
    # --- Reverse String using Stack ---
    def reverse_string(s):
        stack = list(s)
        return ''.join(stack[::-1])
    
    return stack

# ============================================================================
# 7. QUEUES
# ============================================================================
"""
Queue:
- FIFO (First In First Out)
- O(1) enqueue, dequeue
- Use collections.deque (NOT list - list.pop(0) is O(n))
- Applications: BFS, task scheduling, buffering
"""

class Queue:
    """Queue implementation using deque"""
    
    def __init__(self):
        from collections import deque
        self.items = deque()
    
    def enqueue(self, item):
        """Add item to rear - O(1)"""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return front item - O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.popleft()
    
    def front(self):
        """Return front item without removing - O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]
    
    def is_empty(self):
        """Check if queue is empty - O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items - O(1)"""
        return len(self.items)
    
    def __str__(self):
        return str(list(self.items))

def queue_examples():
    """Common queue patterns"""
    
    # --- Using deque as queue ---
    from collections import deque
    queue = deque()
    queue.append(1)      # enqueue
    queue.append(2)
    queue.append(3)
    val = queue.popleft()  # dequeue
    
    # --- Circular Queue / Ring Buffer ---
    class CircularQueue:
        def __init__(self, k):
            self.queue = [None] * k
            self.max_size = k
            self.head = 0
            self.tail = 0
            self.size = 0
        
        def enqueue(self, value):
            if self.is_full():
                return False
            self.queue[self.tail] = value
            self.tail = (self.tail + 1) % self.max_size
            self.size += 1
            return True
        
        def dequeue(self):
            if self.is_empty():
                return False
            self.queue[self.head] = None
            self.head = (self.head + 1) % self.max_size
            self.size -= 1
            return True
        
        def is_empty(self):
            return self.size == 0
        
        def is_full(self):
            return self.size == self.max_size
    
    # --- Priority Queue (using heapq) ---
    import heapq
    pq = []
    heapq.heappush(pq, (2, 'task2'))  # (priority, item)
    heapq.heappush(pq, (1, 'task1'))
    heapq.heappush(pq, (3, 'task3'))
    
    while pq:
        priority, task = heapq.heappop(pq)
        print(f"{task} with priority {priority}")
    
    return queue

# ============================================================================
# 8. LINKED LISTS
# ============================================================================
"""
Linked List:
- Linear data structure with nodes
- Each node contains data and pointer to next node
- O(1) insertion/deletion at beginning
- O(n) search and access by index
- No random access
- Types: Singly, Doubly, Circular
"""

class ListNode:
    """Node for singly linked list"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __str__(self):
        return str(self.val)

class LinkedList:
    """Singly Linked List implementation"""
    
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, val):
        """Add node at end - O(n)"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            self.size += 1
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        self.size += 1
    
    def prepend(self, val):
        """Add node at beginning - O(1)"""
        new_node = ListNode(val, self.head)
        self.head = new_node
        self.size += 1
    
    def insert(self, val, position):
        """Insert at specific position - O(n)"""
        if position == 0:
            self.prepend(val)
            return
        
        new_node = ListNode(val)
        current = self.head
        for _ in range(position - 1):
            if not current:
                raise IndexError("Position out of range")
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
    
    def delete(self, val):
        """Delete first node with value - O(n)"""
        if not self.head:
            return
        
        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return
        
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                self.size -= 1
                return
            current = current.next
    
    def find(self, val):
        """Search for value - O(n)"""
        current = self.head
        position = 0
        while current:
            if current.val == val:
                return position
            current = current.next
            position += 1
        return -1
    
    def reverse(self):
        """Reverse the linked list - O(n)"""
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev
    
    def get_middle(self):
        """Find middle node using slow/fast pointers - O(n)"""
        if not self.head:
            return None
        
        slow = fast = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
    
    def has_cycle(self):
        """Detect cycle using Floyd's algorithm - O(n)"""
        if not self.head:
            return False
        
        slow = fast = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        
        return False
    
    def __str__(self):
        """String representation"""
        values = []
        current = self.head
        while current:
            values.append(str(current.val))
            current = current.next
        return ' -> '.join(values)
    
    def __len__(self):
        return self.size

class DoublyListNode:
    """Node for doubly linked list"""
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class DoublyLinkedList:
    """Doubly Linked List - can traverse both directions"""
    
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, val):
        """Add at end - O(1) with tail pointer"""
        new_node = DoublyListNode(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
    
    def prepend(self, val):
        """Add at beginning - O(1)"""
        new_node = DoublyListNode(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1
    
    def delete(self, val):
        """Delete node - O(n)"""
        current = self.head
        while current:
            if current.val == val:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                
                self.size -= 1
                return
            current = current.next

def linked_list_problems():
    """Common linked list patterns and problems"""
    
    # --- Reverse Linked List ---
    def reverse_list(head):
        prev = None
        current = head
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        return prev
    
    # --- Merge Two Sorted Lists ---
    def merge_two_lists(l1, l2):
        dummy = ListNode(0)
        current = dummy
        
        while l1 and l2:
            if l1.val < l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        current.next = l1 if l1 else l2
        return dummy.next
    
    # --- Remove Nth Node From End ---
    def remove_nth_from_end(head, n):
        dummy = ListNode(0, head)
        fast = slow = dummy
        
        # Move fast n steps ahead
        for _ in range(n):
            fast = fast.next
        
        # Move both until fast reaches end
        while fast.next:
            fast = fast.next
            slow = slow.next
        
        # Remove node
        slow.next = slow.next.next
        return dummy.next
    
    # --- Detect Cycle Start ---
    def detect_cycle_start(head):
        if not head:
            return None
        
        slow = fast = head
        has_cycle = False
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                has_cycle = True
                break
        
        if not has_cycle:
            return None
        
        # Find start of cycle
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return slow
    
    return None

# ============================================================================
# 9. TREES
# ============================================================================
"""
Tree:
- Hierarchical data structure
- Root node, parent-child relationships
- Binary Tree: Each node has at most 2 children
- Applications: File systems, HTML DOM, decision trees
"""

class TreeNode:
    """Node for binary tree"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTree:
    """Binary Tree operations"""
    
    def __init__(self, root=None):
        self.root = root
    
    # --- Tree Traversals ---
    
    def inorder(self, node):
        """Inorder: Left -> Root -> Right - O(n)"""
        if not node:
            return []
        return self.inorder(node.left) + [node.val] + self.inorder(node.right)
    
    def preorder(self, node):
        """Preorder: Root -> Left -> Right - O(n)"""
        if not node:
            return []
        return [node.val] + self.preorder(node.left) + self.preorder(node.right)
    
    def postorder(self, node):
        """Postorder: Left -> Right -> Root - O(n)"""
        if not node:
            return []
        return self.postorder(node.left) + self.postorder(node.right) + [node.val]
    
    def level_order(self, root):
        """Level-order (BFS) traversal - O(n)"""
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level)
        
        return result
    
    # --- Tree Properties ---
    
    def height(self, node):
        """Height of tree - O(n)"""
        if not node:
            return 0
        return 1 + max(self.height(node.left), self.height(node.right))
    
    def size(self, node):
        """Number of nodes - O(n)"""
        if not node:
            return 0
        return 1 + self.size(node.left) + self.size(node.right)
    
    def max_value(self, node):
        """Maximum value in tree - O(n)"""
        if not node:
            return float('-inf')
        return max(node.val, self.max_value(node.left), self.max_value(node.right))
    
    def min_value(self, node):
        """Minimum value in tree - O(n)"""
        if not node:
            return float('inf')
        return min(node.val, self.min_value(node.left), self.min_value(node.right))

class BinarySearchTree:
    """
    Binary Search Tree:
    - Left subtree values < root
    - Right subtree values > root
    - O(log n) average for search, insert, delete
    - O(n) worst case (unbalanced)
    """
    
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        """Insert value - O(log n) average, O(n) worst"""
        if not self.root:
            self.root = TreeNode(val)
        else:
            self._insert_helper(self.root, val)
    
    def _insert_helper(self, node, val):
        if val < node.val:
            if node.left:
                self._insert_helper(node.left, val)
            else:
                node.left = TreeNode(val)
        else:
            if node.right:
                self._insert_helper(node.right, val)
            else:
                node.right = TreeNode(val)
    
    def search(self, val):
        """Search for value - O(log n) average"""
        return self._search_helper(self.root, val)
    
    def _search_helper(self, node, val):
        if not node or node.val == val:
            return node
        
        if val < node.val:
            return self._search_helper(node.left, val)
        return self._search_helper(node.right, val)
    
    def delete(self, val):
        """Delete value - O(log n) average"""
        self.root = self._delete_helper(self.root, val)
    
    def _delete_helper(self, node, val):
        if not node:
            return None
        
        if val < node.val:
            node.left = self._delete_helper(node.left, val)
        elif val > node.val:
            node.right = self._delete_helper(node.right, val)
        else:
            # Node to delete found
            # Case 1: No children
            if not node.left and not node.right:
                return None
            
            # Case 2: One child
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            
            # Case 3: Two children
            # Find inorder successor (min in right subtree)
            min_node = self._find_min(node.right)
            node.val = min_node.val
            node.right = self._delete_helper(node.right, min_node.val)
        
        return node
    
    def _find_min(self, node):
        while node.left:
            node = node.left
        return node
    
    def inorder(self, node):
        """Returns sorted array for BST"""
        if not node:
            return []
        return self.inorder(node.left) + [node.val] + self.inorder(node.right)

def tree_problems():
    """Common tree problems and patterns"""
    
    # --- Maximum Depth ---
    def max_depth(root):
        if not root:
            return 0
        return 1 + max(max_depth(root.left), max_depth(root.right))
    
    # --- Check if Same Tree ---
    def is_same_tree(p, q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        return (p.val == q.val and 
                is_same_tree(p.left, q.left) and 
                is_same_tree(p.right, q.right))
    
    # --- Invert Binary Tree ---
    def invert_tree(root):
        if not root:
            return None
        root.left, root.right = root.right, root.left
        invert_tree(root.left)
        invert_tree(root.right)
        return root
    
    # --- Path Sum ---
    def has_path_sum(root, target_sum):
        if not root:
            return False
        if not root.left and not root.right:
            return root.val == target_sum
        return (has_path_sum(root.left, target_sum - root.val) or
                has_path_sum(root.right, target_sum - root.val))
    
    # --- Lowest Common Ancestor ---
    def lowest_common_ancestor(root, p, q):
        if not root or root == p or root == q:
            return root
        
        left = lowest_common_ancestor(root.left, p, q)
        right = lowest_common_ancestor(root.right, p, q)
        
        if left and right:
            return root
        return left if left else right
    
    # --- Validate BST ---
    def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
        if not root:
            return True
        
        if root.val <= min_val or root.val >= max_val:
            return False
        
        return (is_valid_bst(root.left, min_val, root.val) and
                is_valid_bst(root.right, root.val, max_val))
    
    return None

# ============================================================================
# 10. HEAPS / PRIORITY QUEUES
# ============================================================================
"""
Heap:
- Complete binary tree
- Min Heap: Parent <= Children
- Max Heap: Parent >= Children
- O(log n) insert, delete
- O(1) get min/max
- Python's heapq is a min heap
- Applications: Priority queues, heap sort, median finding
"""

class MinHeap:
    """Min Heap implementation"""
    
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        """Insert element - O(log n)"""
        heapq.heappush(self.heap, val)
    
    def pop(self):
        """Remove and return minimum - O(log n)"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return heapq.heappop(self.heap)
    
    def peek(self):
        """Get minimum without removing - O(1)"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def __len__(self):
        return len(self.heap)

class MaxHeap:
    """Max Heap implementation (using min heap with negation)"""
    
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        """Insert element - O(log n)"""
        heapq.heappush(self.heap, -val)
    
    def pop(self):
        """Remove and return maximum - O(log n)"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return -heapq.heappop(self.heap)
    
    def peek(self):
        """Get maximum without removing - O(1)"""
        if not self.heap:
            raise IndexError("Heap is empty")
        return -self.heap[0]
    
    def __len__(self):
        return len(self.heap)

def heap_examples():
    """Common heap operations and problems"""
    
    # --- Basic Operations ---
    import heapq
    
    # Min Heap
    min_heap = []
    heapq.heappush(min_heap, 3)
    heapq.heappush(min_heap, 1)
    heapq.heappush(min_heap, 4)
    heapq.heappush(min_heap, 2)
    
    min_val = heapq.heappop(min_heap)  # 1
    
    # Convert list to heap (heapify) - O(n)
    arr = [3, 1, 4, 1, 5, 9, 2, 6]
    heapq.heapify(arr)
    
    # Get k smallest/largest
    smallest_3 = heapq.nsmallest(3, arr)
    largest_3 = heapq.nlargest(3, arr)
    
    # --- Kth Largest Element ---
    def find_kth_largest(nums, k):
        """Find kth largest element using min heap"""
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        return heap[0]
    
    # --- Merge K Sorted Lists ---
    def merge_k_sorted_lists(lists):
        """Merge k sorted lists using heap"""
        heap = []
        result = []
        
        # Initialize heap with first element from each list
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst[0], i, 0))
        
        while heap:
            val, list_idx, elem_idx = heapq.heappop(heap)
            result.append(val)
            
            # Add next element from same list
            if elem_idx + 1 < len(lists[list_idx]):
                next_val = lists[list_idx][elem_idx + 1]
                heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
        
        return result
    
    # --- Find Median from Data Stream ---
    class MedianFinder:
        """Maintain median using two heaps"""
        
        def __init__(self):
            self.small = []  # Max heap (lower half)
            self.large = []  # Min heap (upper half)
        
        def add_num(self, num):
            # Add to max heap (small)
            heapq.heappush(self.small, -num)
            
            # Balance: move largest from small to large
            if self.small and self.large and (-self.small[0] > self.large[0]):
                val = -heapq.heappop(self.small)
                heapq.heappush(self.large, val)
            
            # Maintain size property
            if len(self.small) > len(self.large) + 1:
                val = -heapq.heappop(self.small)
                heapq.heappush(self.large, val)
            
            if len(self.large) > len(self.small):
                val = heapq.heappop(self.large)
                heapq.heappush(self.small, -val)
        
        def find_median(self):
            if len(self.small) > len(self.large):
                return -self.small[0]
            return (-self.small[0] + self.large[0]) / 2.0
    
    # --- Top K Frequent Elements ---
    def top_k_frequent(nums, k):
        """Find k most frequent elements"""
        from collections import Counter
        count = Counter(nums)
        return [item for item, freq in count.most_common(k)]
    
    # Or using heap
    def top_k_frequent_heap(nums, k):
        count = Counter(nums)
        return heapq.nlargest(k, count.keys(), key=count.get)
    
    return min_heap

# ============================================================================
# MAIN - Testing Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATA STRUCTURES & ALGORITHMS CHEAT SHEET - PART 1")
    print("=" * 60)
    
    print("\n1. ARRAYS/LISTS")
    print("-" * 60)
    array_operations()
    
    print("\n2. STRINGS")
    print("-" * 60)
    string_operations()
    
    print("\n3. DICTIONARIES")
    print("-" * 60)
    dictionary_operations()
    
    print("\n4. SETS")
    print("-" * 60)
    set_operations()
    
    print("\n5. TUPLES")
    print("-" * 60)
    tuple_operations()
    
    print("\n6. STACKS")
    print("-" * 60)
    stack = Stack()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    print(f"Stack: {stack}")
    print(f"Pop: {stack.pop()}")
    print(f"Peek: {stack.peek()}")
    
    print("\n7. QUEUES")
    print("-" * 60)
    queue = Queue()
    queue.enqueue(1)
    queue.enqueue(2)
    queue.enqueue(3)
    print(f"Queue: {queue}")
    print(f"Dequeue: {queue.dequeue()}")
    print(f"Front: {queue.front()}")
    
    print("\n8. LINKED LISTS")
    print("-" * 60)
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    print(f"Linked List: {ll}")
    ll.reverse()
    print(f"Reversed: {ll}")
    
    print("\n9. BINARY TREES")
    print("-" * 60)
    bt = BinaryTree()
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    bt.root = root
    print(f"Inorder: {bt.inorder(root)}")
    print(f"Preorder: {bt.preorder(root)}")
    print(f"Postorder: {bt.postorder(root)}")
    
    print("\n10. HEAPS")
    print("-" * 60)
    min_heap = MinHeap()
    min_heap.push(3)
    min_heap.push(1)
    min_heap.push(4)
    print(f"Min: {min_heap.peek()}")
    print(f"Pop min: {min_heap.pop()}")
    
    print("\n" + "=" * 60)
    print("Ready for Part 2! Prompt to continue...")
    print("=" * 60)
