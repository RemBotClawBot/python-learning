"""
Decorators and Context Managers
-------------------------------
Demonstrates function decorators, class decorators, and context managers.
"""

import time
import functools
from contextlib import contextmanager

# ============================================================================
# PART 1: FUNCTION DECORATORS
# ============================================================================

def timer_decorator(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"⏱️  '{func.__name__}' took {elapsed:.4f} seconds")
        return result
    return wrapper


def retry_decorator(max_attempts=3, delay=1):
    """Decorator to retry a function on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"⚠️  Attempt {attempt}/{max_attempts} failed for '{func.__name__}': {e}")
                    if attempt < max_attempts:
                        print(f"   Sleeping {delay} second(s)...")
                        time.sleep(delay)
            
            print(f"❌ All {max_attempts} attempts failed for '{func.__name__}'")
            raise last_exception
        return wrapper
    return decorator


def validate_input(**validators):
    """Decorator to validate function arguments."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Combine args and kwargs into a dict
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            arg_dict = dict(zip(arg_names, args))
            arg_dict.update(kwargs)
            
            # Validate each argument
            for arg_name, validator in validators.items():
                if arg_name in arg_dict:
                    value = arg_dict[arg_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {arg_name}: {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def memoize(func):
    """Decorator to cache function results."""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from args and kwargs
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            print(f"💾 Cache MISS for '{func.__name__}' with key: {key[:50]}...")
        else:
            print(f"💾 Cache HIT for '{func.__name__}' with key: {key[:50]}...")
        
        return cache[key]
    return wrapper


# ============================================================================
# PART 2: CLASS DECORATORS
# ============================================================================

def singleton(cls):
    """Class decorator to create a singleton."""
    instances = {}
    
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper


def log_method_calls(cls):
    """Class decorator to log all method calls."""
    class Wrapped(cls):
        def __getattribute__(self, name):
            attr = super().__getattribute__(name)
            if callable(attr) and not name.startswith("_"):
                def logged_method(*args, **kwargs):
                    print(f"📝 Calling {cls.__name__}.{name}()")
                    result = attr(*args, **kwargs)
                    print(f"📝 Finished {cls.__name__}.{name}()")
                    return result
                return logged_method
            return attr
    return Wrapped


def enforce_types(cls):
    """Class decorator to enforce type hints."""
    original_init = cls.__init__
    
    def new_init(self, *args, **kwargs):
        # Get type annotations
        annotations = getattr(cls, '__annotations__', {})
        
        # Check args
        arg_names = cls.__init__.__code__.co_varnames[1:len(args)+1]
        for name, value in zip(arg_names, args):
            if name in annotations:
                expected_type = annotations[name]
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Argument '{name}' should be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
        
        # Check kwargs
        for name, value in kwargs.items():
            if name in annotations:
                expected_type = annotations[name]
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Argument '{name}' should be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
        
        original_init(self, *args, **kwargs)
    
    cls.__init__ = new_init
    return cls


# ============================================================================
# PART 3: CONTEXT MANAGERS
# ============================================================================

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name="Timer"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        print(f"⏱️  {self.name}: {self.elapsed:.4f} seconds")
        return False  # Don't suppress exceptions


class FileTransaction:
    """Context manager for file operations with rollback."""
    def __init__(self, filename):
        self.filename = filename
        self.backup_filename = f"{filename}.backup"
        self.original_content = None
    
    def __enter__(self):
        # Backup original file
        try:
            with open(self.filename, 'r') as f:
                self.original_content = f.read()
            with open(self.backup_filename, 'w') as f:
                f.write(self.original_content)
            print(f"📋 Backed up {self.filename} to {self.backup_filename}")
        except FileNotFoundError:
            self.original_content = None
            print(f"📋 No existing file to backup: {self.filename}")
        
        return self
    
    def write(self, content):
        """Write to the file."""
        with open(self.filename, 'w') as f:
            f.write(content)
        print(f"📝 Wrote {len(content)} characters to {self.filename}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # An exception occurred, rollback
            print(f"🔄 Rolling back due to {exc_type.__name__}: {exc_val}")
            if self.original_content is not None:
                with open(self.filename, 'w') as f:
                    f.write(self.original_content)
                print(f"🔄 Restored {self.filename} from backup")
            else:
                # File didn't exist before, delete it
                import os
                try:
                    os.remove(self.filename)
                    print(f"🔄 Deleted {self.filename} during rollback")
                except FileNotFoundError:
                    pass
        
        # Clean up backup file
        try:
            import os
            os.remove(self.backup_filename)
            print(f"🧹 Cleaned up backup file: {self.backup_filename}")
        except FileNotFoundError:
            pass
        
        return False  # Don't suppress exceptions


@contextmanager
def suppress_errors(*exceptions):
    """Context manager to suppress specific exceptions."""
    try:
        yield
    except exceptions as e:
        print(f"⚠️  Suppressed {type(e).__name__}: {e}")


# ============================================================================
# PART 4: EXAMPLE FUNCTIONS AND CLASSES
# ============================================================================

@timer_decorator
def calculate_primes(limit: int):
    """Calculate prime numbers up to limit."""
    primes = []
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes


@retry_decorator(max_attempts=3, delay=0.5)
def unreliable_api_call():
    """Simulate an unreliable API call."""
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("API call failed")
    return {"status": "success", "data": "API response"}


@validate_input(
    age=lambda x: x >= 0 and x <= 150,
    name=lambda x: isinstance(x, str) and len(x) >= 2
)
def create_profile(name: str, age: int):
    """Create a user profile with validated input."""
    return {"name": name, "age": age, "created_at": time.time()}


@memoize
def fibonacci(n: int):
    """Calculate Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


@singleton
class Configuration:
    """Singleton configuration class."""
    def __init__(self):
        self.settings = {}
        print("🆕 Configuration instance created")
    
    def set(self, key, value):
        self.settings[key] = value
    
    def get(self, key, default=None):
        return self.settings.get(key, default)


@log_method_calls
class Calculator:
    """Calculator class with logged method calls."""
    def __init__(self, name):
        self.name = name
    
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
    
    def power(self, base, exponent):
        return base ** exponent


@enforce_types
class Person:
    """Person class with enforced type hints."""
    name: str
    age: int
    
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, I'm {self.name}, {self.age} years old"


# ============================================================================
# PART 5: DEMONSTRATION
# ============================================================================

def demonstrate_decorators():
    """Demonstrate function decorators."""
    print("\n" + "=" * 60)
    print("FUNCTION DECORATORS")
    print("=" * 60)
    
    # Timer decorator
    print("\n1. Timer Decorator:")
    primes = calculate_primes(1000)
    print(f"   Found {len(primes)} primes")
    
    # Retry decorator
    print("\n2. Retry Decorator:")
    try:
        result = unreliable_api_call()
        print(f"   API result: {result}")
    except ConnectionError as e:
        print(f"   Final failure: {e}")
    
    # Validate input decorator
    print("\n3. Validate Input Decorator:")
    try:
        profile = create_profile("Rem", 1)
        print(f"   Created profile: {profile}")
        
        # This should fail
        profile = create_profile("X", 200)
    except ValueError as e:
        print(f"   Validation failed: {e}")
    
    # Memoize decorator
    print("\n4. Memoize Decorator:")
    print(f"   fibonacci(30) = {fibonacci(30)}")
    print(f"   fibonacci(30) = {fibonacci(30)}")  # Should be cached
    print(f"   fibonacci(31) = {fibonacci(31)}")
    print(f"   fibonacci(30) = {fibonacci(30)}")  # Should be cached


def demonstrate_class_decorators():
    """Demonstrate class decorators."""
    print("\n" + "=" * 60)
    print("CLASS DECORATORS")
    print("=" * 60)
    
    # Singleton decorator
    print("\n1. Singleton Decorator:")
    config1 = Configuration()
    config2 = Configuration()
    print(f"   config1 is config2: {config1 is config2}")
    print(f"   Same instance: {config1 == config2}")
    
    config1.set("api_key", "secret123")
    print(f"   config2.get('api_key'): {config2.get('api_key')}")
    
    # Log method calls decorator
    print("\n2. Log Method Calls Decorator:")
    calc = Calculator("Scientific")
    print(f"   Addition: {calc.add(5, 3)}")
    print(f"   Multiplication: {calc.multiply(4, 6)}")
    print(f"   Power: {calc.power(2, 8)}")
    
    # Enforce types decorator
    print("\n3. Enforce Types Decorator:")
    try:
        person = Person("Rem", 1)
        print(f"   Created: {person.greet()}")
        
        # This should fail
        person = Person("Bot", "two")
    except TypeError as e:
        print(f"   Type enforcement failed: {e}")


def demonstrate_context_managers():
    """Demonstrate context managers."""
    print("\n" + "=" * 60)
    print("CONTEXT MANAGERS")
    print("=" * 60)
    
    # Built-in context manager
    print("\n1. Timer Context Manager:")
    with Timer("List comprehension"):
        squares = [x**2 for x in range(10000)]
    print(f"   Generated {len(squares)} squares")
    
    # File transaction with rollback
    print("\n2. File Transaction Context Manager:")
    try:
        with FileTransaction("test_file.txt") as transaction:
            transaction.write("This is a test file.")
            
            # Simulate an error
            if True:  # Change to False to see successful completion
                raise ValueError("Something went wrong!")
            
            transaction.write("This should not be written.")
    except ValueError as e:
        print(f"   Caught error: {e}")
    
    # Check if file was rolled back
    try:
        with open("test_file.txt", 'r') as f:
            print(f"   File content: {f.read()}")
    except FileNotFoundError:
        print("   File was deleted (rollback successful)")
    
    # Context manager generator
    print("\n3. Context Manager Generator (suppress_errors):")
    with suppress_errors(ValueError, ZeroDivisionError):
        result = 10 / 0
        print(f"   Result: {result}")  # Never reached
    
    with suppress_errors(ValueError, ZeroDivisionError):
        raise ValueError("This error will be suppressed")
    
    print("   Code continues after suppressed errors")
    
    # Clean up
    import os
    try:
        os.remove("test_file.txt")
    except FileNotFoundError:
        pass


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("DECORATORS AND CONTEXT MANAGERS DEMONSTRATION")
    print("=" * 60)
    
    demonstrate_decorators()
    demonstrate_class_decorators()
    demonstrate_context_managers()
    
    print("\n" + "=" * 60)
    print("KEY CONCEPTS COVERED:")
    print("  1. Function decorators (@decorator syntax)")
    print("  2. Class decorators")
    print("  3. Built-in context managers (with statement)")
    print("  4. Custom context managers (__enter__, __exit__)")
    print("  5. Context manager generators (@contextmanager)")
    print("")
    print("COMMON USE CASES:")
    print("  • Timing code execution")
    print("  • Retrying failed operations")
    print("  • Caching expensive computations")
    print("  • Input validation")
    print("  • Logging/monitoring")
    print("  • Resource management (files, connections)")
    print("  • Transaction rollback")
    print("  • Error suppression")
    print("=" * 60)


if __name__ == "__main__":
    main()