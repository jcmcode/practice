"""
PYTHON OBJECTS - A COMPREHENSIVE GUIDE
=======================================
This file demonstrates how to create and use objects in Python.
Objects are instances of classes that encapsulate data (attributes) and behavior (methods).
"""

# =============================================================================
# 1. BASIC CLASS DEFINITION
# =============================================================================

class Dog:
    """
    A simple class representing a dog.
    Classes are blueprints for creating objects.
    """
    
    def __init__(self, name, age):
        """
        Constructor method - called when creating a new object.
        'self' refers to the instance being created.
        
        Args:
            name: The dog's name
            age: The dog's age
        """
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute
    
    def bark(self):
        """Instance method - behavior of the object"""
        return f"{self.name} says Woof!"
    
    def get_info(self):
        """Return information about the dog"""
        return f"{self.name} is {self.age} years old"


# Creating objects (instances of the Dog class)
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

print(dog1.bark())          # Buddy says Woof!
print(dog2.get_info())      # Max is 5 years old


# =============================================================================
# 2. CLASS ATTRIBUTES vs INSTANCE ATTRIBUTES
# =============================================================================

class Cat:
    """Demonstrates class attributes vs instance attributes"""
    
    # Class attribute - shared by all instances
    species = "Felis catus"
    count = 0  # Track number of cats created
    
    def __init__(self, name, color):
        # Instance attributes - unique to each instance
        self.name = name
        self.color = color
        Cat.count += 1  # Increment class attribute
    
    def describe(self):
        return f"{self.name} is a {self.color} cat of species {Cat.species}"


cat1 = Cat("Whiskers", "orange")
cat2 = Cat("Shadow", "black")

print(cat1.describe())      # Whiskers is a orange cat of species Felis catus
print(f"Total cats: {Cat.count}")  # Total cats: 2


# =============================================================================
# 3. METHODS: INSTANCE, CLASS, AND STATIC
# =============================================================================

class MathOperations:
    """Demonstrates different types of methods"""
    
    pi = 3.14159  # Class attribute
    
    def __init__(self, value):
        self.value = value
    
    # Instance method - operates on instance data
    def square(self):
        """Returns the square of the instance value"""
        return self.value ** 2
    
    # Class method - operates on class data
    @classmethod
    def circle_area(cls, radius):
        """
        Calculates circle area using class attribute pi.
        Takes 'cls' instead of 'self' - refers to the class itself.
        """
        return cls.pi * radius ** 2
    
    # Static method - doesn't access instance or class data
    @staticmethod
    def add(a, b):
        """
        Simple utility function that doesn't need instance or class data.
        Can be called on the class or an instance.
        """
        return a + b


math_obj = MathOperations(5)
print(math_obj.square())                    # 25 (instance method)
print(MathOperations.circle_area(10))       # 314.159 (class method)
print(MathOperations.add(3, 7))             # 10 (static method)


# =============================================================================
# 4. ENCAPSULATION - PUBLIC, PROTECTED, AND PRIVATE ATTRIBUTES
# =============================================================================

class BankAccount:
    """Demonstrates encapsulation and access control"""
    
    def __init__(self, owner, balance):
        self.owner = owner              # Public attribute
        self._account_number = "12345"  # Protected attribute (convention: single underscore)
        self.__balance = balance        # Private attribute (name mangling: double underscore)
    
    # Getter method for private attribute
    def get_balance(self):
        return self.__balance
    
    # Setter method with validation
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return f"Deposited ${amount}. New balance: ${self.__balance}"
        return "Invalid deposit amount"
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return f"Withdrew ${amount}. New balance: ${self.__balance}"
        return "Invalid withdrawal amount"


account = BankAccount("John", 1000)
print(account.owner)                    # John (public - OK)
print(account.get_balance())            # 1000 (accessing private via method)
print(account.deposit(500))             # Deposited $500. New balance: $1500
# print(account.__balance)              # AttributeError - private attribute


# =============================================================================
# 5. PROPERTY DECORATORS - PYTHONIC GETTERS AND SETTERS
# =============================================================================

class Person:
    """Demonstrates property decorators for controlled attribute access"""
    
    def __init__(self, name, age):
        self._name = name
        self._age = age
    
    @property
    def name(self):
        """Getter for name - accessed like an attribute"""
        return self._name.title()
    
    @name.setter
    def name(self, value):
        """Setter for name with validation"""
        if isinstance(value, str) and len(value) > 0:
            self._name = value
        else:
            raise ValueError("Name must be a non-empty string")
    
    @property
    def age(self):
        """Getter for age"""
        return self._age
    
    @age.setter
    def age(self, value):
        """Setter for age with validation"""
        if isinstance(value, int) and value >= 0:
            self._age = value
        else:
            raise ValueError("Age must be a non-negative integer")
    
    @property
    def is_adult(self):
        """Read-only computed property"""
        return self._age >= 18


person = Person("alice", 25)
print(person.name)          # Alice (property getter)
person.age = 30             # Using property setter
print(person.is_adult)      # True (read-only computed property)


# =============================================================================
# 6. INHERITANCE - CREATING SUBCLASSES
# =============================================================================

class Animal:
    """Parent class (base class)"""
    
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        return "Some generic sound"
    
    def info(self):
        return f"{self.name} is a {self.species}"


class Bird(Animal):
    """Child class (derived class) inheriting from Animal"""
    
    def __init__(self, name, can_fly=True):
        # Call parent class constructor
        super().__init__(name, "Bird")
        self.can_fly = can_fly
    
    # Override parent method
    def make_sound(self):
        return "Chirp chirp!"
    
    # Add new method specific to Bird
    def fly(self):
        if self.can_fly:
            return f"{self.name} is flying!"
        return f"{self.name} cannot fly"


class Fish(Animal):
    """Another child class inheriting from Animal"""
    
    def __init__(self, name, water_type="freshwater"):
        super().__init__(name, "Fish")
        self.water_type = water_type
    
    def make_sound(self):
        return "Blub blub"
    
    def swim(self):
        return f"{self.name} is swimming in {self.water_type}"


# Using inheritance
bird = Bird("Tweety")
fish = Fish("Nemo", "saltwater")

print(bird.info())          # Tweety is a Bird (inherited method)
print(bird.make_sound())    # Chirp chirp! (overridden method)
print(bird.fly())           # Tweety is flying! (new method)
print(fish.swim())          # Nemo is swimming in saltwater


# =============================================================================
# 7. SPECIAL METHODS (MAGIC METHODS / DUNDER METHODS)
# =============================================================================

class Book:
    """Demonstrates special methods for operator overloading"""
    
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
    
    # String representation for developers (used by repr())
    def __repr__(self):
        return f"Book('{self.title}', '{self.author}', {self.pages})"
    
    # String representation for users (used by print() and str())
    def __str__(self):
        return f"'{self.title}' by {self.author}"
    
    # Length of the object
    def __len__(self):
        return self.pages
    
    # Comparison operators
    def __eq__(self, other):
        """Equality comparison"""
        return self.pages == other.pages
    
    def __lt__(self, other):
        """Less than comparison"""
        return self.pages < other.pages
    
    # Addition operator
    def __add__(self, other):
        """Combine two books"""
        return self.pages + other.pages


book1 = Book("1984", "George Orwell", 328)
book2 = Book("Animal Farm", "George Orwell", 112)

print(book1)                    # '1984' by George Orwell (__str__)
print(repr(book1))              # Book('1984', 'George Orwell', 328) (__repr__)
print(len(book1))               # 328 (__len__)
print(book1 > book2)            # True (__lt__ and __gt__)
print(book1 + book2)            # 440 (__add__)


# =============================================================================
# 8. COMPOSITION - "HAS-A" RELATIONSHIP
# =============================================================================

class Engine:
    """Component class"""
    
    def __init__(self, horsepower):
        self.horsepower = horsepower
    
    def start(self):
        return f"Engine with {self.horsepower}hp started"


class Car:
    """Class that uses composition"""
    
    def __init__(self, brand, model, horsepower):
        self.brand = brand
        self.model = model
        self.engine = Engine(horsepower)  # Car HAS-A Engine
    
    def start(self):
        return f"{self.brand} {self.model}: {self.engine.start()}"


car = Car("Toyota", "Camry", 200)
print(car.start())  # Toyota Camry: Engine with 200hp started


# =============================================================================
# 9. CLASS WITH MULTIPLE CONSTRUCTORS (USING CLASS METHODS)
# =============================================================================

class Date:
    """Demonstrates multiple ways to create objects"""
    
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    @classmethod
    def from_string(cls, date_string):
        """Alternative constructor from string format 'YYYY-MM-DD'"""
        year, month, day = map(int, date_string.split('-'))
        return cls(year, month, day)
    
    @classmethod
    def today(cls):
        """Alternative constructor for today's date"""
        import datetime
        today = datetime.date.today()
        return cls(today.year, today.month, today.day)
    
    def __str__(self):
        return f"{self.year}-{self.month:02d}-{self.day:02d}"


# Different ways to create Date objects
date1 = Date(2025, 12, 31)              # Regular constructor
date2 = Date.from_string("2025-12-31")  # From string
date3 = Date.today()                     # Today's date

print(date1)  # 2025-12-31


# =============================================================================
# 10. REAL-WORLD EXAMPLE - COMPLETE CLASS
# =============================================================================

class ShoppingCart:
    """A complete example demonstrating multiple OOP concepts"""
    
    def __init__(self, owner):
        self.owner = owner
        self._items = []  # Protected attribute
        self._discount = 0
    
    def add_item(self, item, price, quantity=1):
        """Add an item to the cart"""
        self._items.append({
            'item': item,
            'price': price,
            'quantity': quantity
        })
        return f"Added {quantity} x {item} to cart"
    
    def remove_item(self, item):
        """Remove an item from the cart"""
        self._items = [i for i in self._items if i['item'] != item]
        return f"Removed {item} from cart"
    
    @property
    def subtotal(self):
        """Calculate subtotal before discount"""
        return sum(item['price'] * item['quantity'] for item in self._items)
    
    @property
    def total(self):
        """Calculate total after discount"""
        return self.subtotal * (1 - self._discount)
    
    def apply_discount(self, percentage):
        """Apply discount to the cart"""
        if 0 <= percentage <= 100:
            self._discount = percentage / 100
            return f"Applied {percentage}% discount"
        return "Invalid discount percentage"
    
    def __len__(self):
        """Return number of items in cart"""
        return len(self._items)
    
    def __str__(self):
        """String representation of the cart"""
        items_str = '\n'.join([f"  {i['quantity']} x {i['item']}: ${i['price']}" 
                               for i in self._items])
        return f"{self.owner}'s Cart:\n{items_str}\nTotal: ${self.total:.2f}"


# Using the ShoppingCart class
cart = ShoppingCart("Alice")
cart.add_item("Apple", 1.50, 3)
cart.add_item("Banana", 0.75, 5)
cart.add_item("Milk", 3.99, 1)
cart.apply_discount(10)

print(cart)
print(f"\nItems in cart: {len(cart)}")


# =============================================================================
# KEY TAKEAWAYS
# =============================================================================
"""
1. Classes are blueprints; objects are instances of classes
2. __init__ is the constructor method
3. self refers to the instance
4. Instance attributes are unique to each object
5. Class attributes are shared by all instances
6. Methods are functions that belong to a class
7. Inheritance allows code reuse (use super() to call parent methods)
8. Encapsulation: use _ for protected, __ for private attributes
9. Properties provide controlled access to attributes
10. Special methods (__str__, __len__, etc.) enable Pythonic behavior
"""
