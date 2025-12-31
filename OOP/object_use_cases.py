"""
OBJECT USE CASES IN PYTHON
===========================
This file explores practical use cases for objects and when to use OOP vs other approaches.
Understanding when and why to use objects is key to writing maintainable, scalable code.
"""

# =============================================================================
# WHY USE OBJECTS? WHEN ARE THEY USEFUL?
# =============================================================================
"""
Objects are useful when you need to:
1. Model real-world entities (users, products, accounts)
2. Bundle related data and behavior together
3. Maintain state across multiple operations
4. Create reusable, modular code
5. Implement complex systems with multiple interacting components
6. Encapsulate implementation details
7. Use inheritance to share common functionality
"""


# =============================================================================
# USE CASE 1: MODELING REAL-WORLD ENTITIES
# =============================================================================
"""
Objects excel at representing things from the real world with their 
properties and behaviors.
"""

class Student:
    """Model a student with grades, enrollment, and academic operations"""
    
    def __init__(self, student_id, name, major):
        self.student_id = student_id
        self.name = name
        self.major = major
        self.grades = {}  # {course: grade}
        self.enrolled_courses = []
    
    def enroll(self, course):
        """Enroll in a course"""
        if course not in self.enrolled_courses:
            self.enrolled_courses.append(course)
            return f"{self.name} enrolled in {course}"
        return f"Already enrolled in {course}"
    
    def add_grade(self, course, grade):
        """Record a grade for a course"""
        if course in self.enrolled_courses:
            self.grades[course] = grade
            return f"Grade {grade} recorded for {course}"
        return f"Not enrolled in {course}"
    
    def get_gpa(self):
        """Calculate GPA"""
        if not self.grades:
            return 0.0
        
        grade_points = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
        total = sum(grade_points.get(grade, 0) for grade in self.grades.values())
        return round(total / len(self.grades), 2)
    
    def __str__(self):
        return f"{self.name} ({self.student_id}) - {self.major} - GPA: {self.get_gpa()}"


# Example usage
student = Student("S12345", "Emma Wilson", "Computer Science")
student.enroll("Data Structures")
student.enroll("Algorithms")
student.add_grade("Data Structures", "A")
student.add_grade("Algorithms", "B")
print(student)  # Emma Wilson (S12345) - Computer Science - GPA: 3.5


# =============================================================================
# USE CASE 2: MAINTAINING STATE ACROSS OPERATIONS
# =============================================================================
"""
Objects are perfect for scenarios where you need to maintain state
between multiple operations, rather than passing data around.
"""

class TextProcessor:
    """Process text while maintaining statistics and history"""
    
    def __init__(self):
        self.text = ""
        self.word_count = 0
        self.operations_history = []
    
    def add_text(self, text):
        """Add text to the processor"""
        self.text += text + " "
        self.word_count = len(self.text.split())
        self.operations_history.append(f"Added text: {text[:20]}...")
    
    def to_uppercase(self):
        """Convert text to uppercase"""
        self.text = self.text.upper()
        self.operations_history.append("Converted to uppercase")
    
    def remove_word(self, word):
        """Remove all instances of a word"""
        self.text = self.text.replace(word, "")
        self.word_count = len(self.text.split())
        self.operations_history.append(f"Removed word: {word}")
    
    def get_stats(self):
        """Get statistics about the text"""
        return {
            'total_words': self.word_count,
            'total_chars': len(self.text),
            'operations': len(self.operations_history)
        }
    
    def get_history(self):
        """Get operation history"""
        return self.operations_history


# Example usage
processor = TextProcessor()
processor.add_text("Hello world")
processor.add_text("Python is awesome")
processor.to_uppercase()
print(processor.text)  # HELLO WORLD PYTHON IS AWESOME
print(processor.get_stats())  # Statistics maintained throughout


# =============================================================================
# USE CASE 3: GAME DEVELOPMENT
# =============================================================================
"""
Games are a classic use case for OOP - characters, items, and environments
are naturally represented as objects with state and behavior.
"""

class Character:
    """Base character class for a game"""
    
    def __init__(self, name, health, attack_power):
        self.name = name
        self.max_health = health
        self.health = health
        self.attack_power = attack_power
        self.inventory = []
        self.is_alive = True
    
    def take_damage(self, damage):
        """Reduce health when damaged"""
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            return f"{self.name} has been defeated!"
        return f"{self.name} took {damage} damage. HP: {self.health}/{self.max_health}"
    
    def heal(self, amount):
        """Restore health"""
        self.health = min(self.health + amount, self.max_health)
        return f"{self.name} healed {amount} HP. Current HP: {self.health}/{self.max_health}"
    
    def attack(self, target):
        """Attack another character"""
        if not self.is_alive:
            return f"{self.name} cannot attack (defeated)"
        
        return target.take_damage(self.attack_power)
    
    def add_to_inventory(self, item):
        """Add item to inventory"""
        self.inventory.append(item)
        return f"{self.name} picked up {item}"


class Warrior(Character):
    """Warrior character with special abilities"""
    
    def __init__(self, name):
        super().__init__(name, health=150, attack_power=25)
        self.rage = 0
    
    def power_attack(self, target):
        """Special attack that uses rage"""
        if self.rage >= 50:
            damage = self.attack_power * 2
            self.rage = 0
            target.take_damage(damage)
            return f"{self.name} used Power Attack! {damage} damage dealt!"
        return f"Not enough rage (need 50, have {self.rage})"
    
    def take_damage(self, damage):
        """Warrior gains rage when damaged"""
        self.rage = min(self.rage + 10, 100)
        return super().take_damage(damage)


class Mage(Character):
    """Mage character with mana"""
    
    def __init__(self, name):
        super().__init__(name, health=80, attack_power=15)
        self.mana = 100
        self.max_mana = 100
    
    def cast_fireball(self, target):
        """Cast a fireball spell"""
        mana_cost = 30
        if self.mana >= mana_cost:
            self.mana -= mana_cost
            damage = 40
            target.take_damage(damage)
            return f"{self.name} cast Fireball! {damage} damage! Mana: {self.mana}/{self.max_mana}"
        return f"Not enough mana (need {mana_cost}, have {self.mana})"


# Example game battle
warrior = Warrior("Conan")
mage = Mage("Gandalf")

print(mage.cast_fireball(warrior))  # Mage attacks warrior
print(warrior.attack(mage))         # Warrior attacks back
print(warrior.power_attack(mage))   # Not enough rage yet


# =============================================================================
# USE CASE 4: DATA MANAGEMENT AND VALIDATION
# =============================================================================
"""
Objects are great for managing data with validation rules and 
ensuring data integrity.
"""

class Email:
    """Email validator and formatter"""
    
    def __init__(self, address):
        self.address = self._validate(address)
    
    def _validate(self, address):
        """Validate email format"""
        if '@' not in address or '.' not in address.split('@')[1]:
            raise ValueError(f"Invalid email format: {address}")
        return address.lower().strip()
    
    @property
    def username(self):
        """Extract username from email"""
        return self.address.split('@')[0]
    
    @property
    def domain(self):
        """Extract domain from email"""
        return self.address.split('@')[1]
    
    def __str__(self):
        return self.address
    
    def __eq__(self, other):
        return self.address == other.address


class PhoneNumber:
    """Phone number validator and formatter"""
    
    def __init__(self, number):
        self.raw_number = number
        self.formatted_number = self._format(self._clean(number))
    
    def _clean(self, number):
        """Remove non-digit characters"""
        return ''.join(filter(str.isdigit, number))
    
    def _format(self, number):
        """Format as (XXX) XXX-XXXX"""
        if len(number) != 10:
            raise ValueError(f"Phone number must be 10 digits, got {len(number)}")
        return f"({number[:3]}) {number[3:6]}-{number[6:]}"
    
    def __str__(self):
        return self.formatted_number


class Contact:
    """Contact information with validation"""
    
    def __init__(self, name, email, phone):
        self.name = name
        self.email = Email(email)  # Validates on creation
        self.phone = PhoneNumber(phone)  # Validates on creation
    
    def __str__(self):
        return f"{self.name}\nEmail: {self.email}\nPhone: {self.phone}"


# Example usage - validation happens automatically
contact = Contact("John Doe", "john.doe@example.com", "555-123-4567")
print(contact)
# Formatted output:
# John Doe
# Email: john.doe@example.com
# Phone: (555) 123-4567


# =============================================================================
# USE CASE 5: API CLIENTS AND NETWORK REQUESTS
# =============================================================================
"""
Objects are ideal for encapsulating API interactions, managing
authentication, and handling requests.
"""

class APIClient:
    """Simulated API client for a web service"""
    
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.session_active = False
        self.request_count = 0
    
    def authenticate(self):
        """Simulate authentication"""
        # In real scenario, this would make an HTTP request
        self.session_active = True
        return "Authentication successful"
    
    def _make_request(self, endpoint, method="GET", data=None):
        """Simulate making an API request"""
        if not self.session_active:
            return "Error: Not authenticated"
        
        self.request_count += 1
        # Simulate request
        url = f"{self.base_url}/{endpoint}"
        return f"[{method}] {url} - Request #{self.request_count}"
    
    def get_user(self, user_id):
        """Get user information"""
        return self._make_request(f"users/{user_id}")
    
    def create_post(self, title, content):
        """Create a new post"""
        data = {'title': title, 'content': content}
        return self._make_request("posts", method="POST", data=data)
    
    def get_stats(self):
        """Get client statistics"""
        return {
            'requests_made': self.request_count,
            'session_active': self.session_active
        }


# Example usage
api = APIClient("https://api.example.com", "secret_key_123")
api.authenticate()
print(api.get_user(42))
print(api.create_post("Hello", "World"))
print(api.get_stats())


# =============================================================================
# USE CASE 6: CONFIGURATION MANAGEMENT
# =============================================================================
"""
Objects can manage application configuration with defaults, validation,
and environment-specific settings.
"""

class AppConfig:
    """Application configuration manager"""
    
    # Default configuration
    DEFAULTS = {
        'debug': False,
        'max_connections': 100,
        'timeout': 30,
        'log_level': 'INFO'
    }
    
    def __init__(self, environment='development'):
        self.environment = environment
        self.settings = self.DEFAULTS.copy()
        self._load_environment_config()
    
    def _load_environment_config(self):
        """Load environment-specific settings"""
        if self.environment == 'development':
            self.settings.update({
                'debug': True,
                'log_level': 'DEBUG'
            })
        elif self.environment == 'production':
            self.settings.update({
                'debug': False,
                'max_connections': 500,
                'log_level': 'WARNING'
            })
    
    def get(self, key, default=None):
        """Get a configuration value"""
        return self.settings.get(key, default)
    
    def set(self, key, value):
        """Set a configuration value"""
        self.settings[key] = value
    
    def is_debug(self):
        """Check if debug mode is enabled"""
        return self.settings.get('debug', False)
    
    def __str__(self):
        return f"Config[{self.environment}]: {self.settings}"


# Example usage
dev_config = AppConfig('development')
prod_config = AppConfig('production')

print(f"Dev Debug: {dev_config.is_debug()}")    # True
print(f"Prod Debug: {prod_config.is_debug()}")  # False
print(f"Prod Max Connections: {prod_config.get('max_connections')}")  # 500


# =============================================================================
# USE CASE 7: FILE HANDLING AND DATA PROCESSING
# =============================================================================
"""
Objects can manage file operations while maintaining state,
buffering, and error handling.
"""

class CSVProcessor:
    """Process CSV files with statistics tracking"""
    
    def __init__(self, delimiter=','):
        self.delimiter = delimiter
        self.rows_processed = 0
        self.headers = []
        self.data = []
        self.errors = []
    
    def load_from_string(self, csv_string):
        """Load CSV from a string (simulated file reading)"""
        lines = csv_string.strip().split('\n')
        
        if not lines:
            self.errors.append("Empty CSV data")
            return
        
        # First line is headers
        self.headers = lines[0].split(self.delimiter)
        
        # Process data rows
        for i, line in enumerate(lines[1:], start=2):
            try:
                values = line.split(self.delimiter)
                if len(values) == len(self.headers):
                    row_dict = dict(zip(self.headers, values))
                    self.data.append(row_dict)
                    self.rows_processed += 1
                else:
                    self.errors.append(f"Line {i}: Column count mismatch")
            except Exception as e:
                self.errors.append(f"Line {i}: {str(e)}")
    
    def filter_by(self, column, value):
        """Filter rows by column value"""
        if column not in self.headers:
            return []
        return [row for row in self.data if row.get(column) == value]
    
    def get_column(self, column_name):
        """Extract all values from a column"""
        if column_name not in self.headers:
            return []
        return [row[column_name] for row in self.data]
    
    def get_stats(self):
        """Get processing statistics"""
        return {
            'rows_processed': self.rows_processed,
            'columns': len(self.headers),
            'errors': len(self.errors)
        }


# Example usage
csv_data = """name,age,city
Alice,25,New York
Bob,30,San Francisco
Charlie,35,New York"""

processor = CSVProcessor()
processor.load_from_string(csv_data)
print(f"Headers: {processor.headers}")
print(f"New York residents: {processor.filter_by('city', 'New York')}")
print(f"Stats: {processor.get_stats()}")


# =============================================================================
# USE CASE 8: CACHING AND MEMOIZATION
# =============================================================================
"""
Objects can maintain caches to improve performance by storing
previously computed results.
"""

class Calculator:
    """Calculator with result caching"""
    
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def fibonacci(self, n):
        """Calculate Fibonacci number with caching"""
        # Check cache
        if n in self.cache:
            self.cache_hits += 1
            return self.cache[n]
        
        # Calculate
        self.cache_misses += 1
        if n <= 1:
            result = n
        else:
            result = self.fibonacci(n - 1) + self.fibonacci(n - 2)
        
        # Store in cache
        self.cache[n] = result
        return result
    
    def factorial(self, n):
        """Calculate factorial with caching"""
        cache_key = f"factorial_{n}"
        
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        if n <= 1:
            result = 1
        else:
            result = n * self.factorial(n - 1)
        
        self.cache[cache_key] = result
        return result
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_size': len(self.cache),
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }


# Example usage
calc = Calculator()
print(calc.fibonacci(10))  # First call - computes
print(calc.fibonacci(10))  # Second call - cached
print(calc.factorial(5))   # 120
print(calc.get_cache_stats())


# =============================================================================
# USE CASE 9: STATE MACHINES
# =============================================================================
"""
Objects are perfect for implementing state machines where behavior
changes based on current state.
"""

class TrafficLight:
    """Traffic light state machine"""
    
    STATES = ['RED', 'YELLOW', 'GREEN']
    
    def __init__(self):
        self.current_state = 'RED'
        self.state_history = ['RED']
        self.cycle_count = 0
    
    def next_state(self):
        """Transition to the next state"""
        state_transitions = {
            'RED': 'GREEN',
            'GREEN': 'YELLOW',
            'YELLOW': 'RED'
        }
        
        self.current_state = state_transitions[self.current_state]
        self.state_history.append(self.current_state)
        
        if self.current_state == 'RED':
            self.cycle_count += 1
        
        return self.current_state
    
    def can_go(self):
        """Check if traffic can proceed"""
        return self.current_state == 'GREEN'
    
    def should_prepare_to_stop(self):
        """Check if drivers should prepare to stop"""
        return self.current_state == 'YELLOW'
    
    def get_status(self):
        """Get current status"""
        return {
            'state': self.current_state,
            'can_go': self.can_go(),
            'cycles_completed': self.cycle_count
        }


# Example usage
light = TrafficLight()
print(f"Initial: {light.current_state}")  # RED
print(f"Can go? {light.can_go()}")        # False

light.next_state()
print(f"After change: {light.current_state}")  # GREEN
print(f"Can go? {light.can_go()}")              # True


# =============================================================================
# USE CASE 10: BUILDER PATTERN FOR COMPLEX OBJECTS
# =============================================================================
"""
When creating complex objects with many optional parameters,
the builder pattern using objects makes code more readable.
"""

class Pizza:
    """Pizza product"""
    
    def __init__(self):
        self.size = None
        self.crust = None
        self.toppings = []
        self.cheese = None
        self.sauce = None
    
    def __str__(self):
        toppings_str = ", ".join(self.toppings) if self.toppings else "none"
        return f"{self.size} {self.crust} crust pizza with {self.cheese} cheese, {self.sauce} sauce, and toppings: {toppings_str}"


class PizzaBuilder:
    """Builder for creating pizzas"""
    
    def __init__(self):
        self.pizza = Pizza()
    
    def set_size(self, size):
        """Set pizza size"""
        self.pizza.size = size
        return self  # Return self for method chaining
    
    def set_crust(self, crust):
        """Set crust type"""
        self.pizza.crust = crust
        return self
    
    def set_cheese(self, cheese):
        """Set cheese type"""
        self.pizza.cheese = cheese
        return self
    
    def set_sauce(self, sauce):
        """Set sauce type"""
        self.pizza.sauce = sauce
        return self
    
    def add_topping(self, topping):
        """Add a topping"""
        self.pizza.toppings.append(topping)
        return self
    
    def build(self):
        """Return the completed pizza"""
        # Set defaults if not specified
        if not self.pizza.size:
            self.pizza.size = "medium"
        if not self.pizza.crust:
            self.pizza.crust = "regular"
        if not self.pizza.cheese:
            self.pizza.cheese = "mozzarella"
        if not self.pizza.sauce:
            self.pizza.sauce = "tomato"
        
        return self.pizza


# Example usage - method chaining makes it very readable
pizza = (PizzaBuilder()
         .set_size("large")
         .set_crust("thin")
         .add_topping("pepperoni")
         .add_topping("mushrooms")
         .add_topping("olives")
         .set_cheese("extra mozzarella")
         .build())

print(pizza)


# =============================================================================
# WHEN NOT TO USE OBJECTS
# =============================================================================
"""
Not everything needs to be an object! Avoid over-engineering.

DON'T use objects when:
1. Simple functions would suffice (pure functions with no state)
2. You only need a single operation with no related data
3. You're just grouping unrelated utilities
4. The object would have no state (use functions or modules instead)

GOOD use of functions instead of objects:
"""

# Simple utility functions - no need for a class
def calculate_tax(price, tax_rate):
    """Simple calculation - function is better than object"""
    return price * tax_rate

def format_currency(amount):
    """Simple formatting - function is sufficient"""
    return f"${amount:,.2f}"

# These don't need state, so functions are cleaner than objects


# =============================================================================
# SUMMARY OF OBJECT USE CASES
# =============================================================================
"""
Objects are best for:
1. ✅ Modeling real-world entities (Student, Car, Account)
2. ✅ Maintaining state across operations (TextProcessor, Calculator)
3. ✅ Complex systems with interacting components (Games, APIs)
4. ✅ Data validation and management (Email, PhoneNumber)
5. ✅ Encapsulating related behavior and data (APIClient, Config)
6. ✅ Inheritance when multiple types share behavior (Character types)
7. ✅ State machines (TrafficLight, Order status)
8. ✅ Caching and performance optimization
9. ✅ Builder patterns for complex construction
10. ✅ Managing resources (files, connections, sessions)

Avoid objects for:
❌ Simple, stateless operations (use functions)
❌ One-off calculations
❌ Collections of unrelated utilities
❌ When dictionaries or namedtuples would be clearer
"""

# =============================================================================
# QUICK DECISION GUIDE
# =============================================================================
"""
Ask yourself:
- Does this have state that persists between operations? → Use an object
- Does this model something from the real world? → Use an object
- Do I need multiple instances with different data? → Use an object
- Is this just a simple calculation or transformation? → Use a function
- Am I just grouping related constants? → Use a module or dict
- Do I need inheritance or polymorphism? → Use objects
"""
