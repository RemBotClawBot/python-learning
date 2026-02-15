"""
Error Handling and Exception Examples
--------------------------------------
Demonstrates try/except/finally, custom exceptions, and error patterns.
"""

def basic_error_handling():
    """Demonstrate basic try/except/else/finally."""
    print("=" * 60)
    print("BASIC ERROR HANDLING")
    print("=" * 60)
    
    # Example 1: Basic try/except
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        print(f"1. Caught ZeroDivisionError: {e}")
    
    # Example 2: Multiple exception types
    try:
        user_input = input("Enter a number: ")
        number = int(user_input)
        result = 100 / number
        print(f"Result: {result}")
    except ValueError:
        print("2. Please enter a valid integer!")
    except ZeroDivisionError:
        print("2. Cannot divide by zero!")
    except Exception as e:
        print(f"2. Unexpected error: {e}")
    
    # Example 3: try/except/else/finally
    file = None
    try:
        file = open("example_data.txt", "r")
        content = file.read()
        print(f"3. File content length: {len(content)} characters")
    except FileNotFoundError:
        print("3. File not found, creating one...")
        content = "Sample data\n"
        with open("example_data.txt", "w") as f:
            f.write(content)
    except IOError as e:
        print(f"3. IO Error: {e}")
    else:
        print("3. File read successfully!")
    finally:
        if file and not file.closed:
            file.close()
            print("3. File closed in finally block")
        print("3. Finally block always executes")
    
    print()


def handle_mixed_errors():
    """Demonstrate handling multiple potential errors."""
    print("=" * 60)
    print("HANDLING MULTIPLE POTENTIAL ERRORS")
    print("=" * 60)
    
    data_sources = [
        {"type": "list", "data": [1, 2, 3, 4, 5]},
        {"type": "dict", "data": {"key": "value"}},
        {"type": "int", "data": 42},
        {"type": "string", "data": "hello"},
        {"type": "none", "data": None},
    ]
    
    for i, source in enumerate(data_sources, 1):
        data = source["data"]
        print(f"\nSource {i} ({source['type']}): {data}")
        
        try:
            # Try various operations that might fail
            if source["type"] == "list":
                print(f"  First element: {data[0]}")
                print(f"  Last element: {data[-1]}")
                print(f"  Sum: {sum(data)}")
            
            elif source["type"] == "dict":
                print(f"  Value for 'key': {data['key']}")
                print(f"  Keys: {list(data.keys())}")
            
            elif source["type"] == "int":
                print(f"  Square root: {data ** 0.5}")
                print(f"  As string: {str(data)[::-1]}")
            
            elif source["type"] == "string":
                print(f"  Uppercase: {data.upper()}")
                print(f"  First character: {data[0]}")
                print(f"  Length: {len(data)}")
            
            elif source["type"] == "none":
                # This will raise AttributeError
                print(f"  Uppercase: {data.upper()}")
        
        except TypeError as e:
            print(f"  ❌ TypeError: {e}")
        except IndexError as e:
            print(f"  ❌ IndexError: {e}")
        except KeyError as e:
            print(f"  ❌ KeyError: {e}")
        except AttributeError as e:
            print(f"  ❌ AttributeError: {e}")
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
        else:
            print(f"  ✅ All operations succeeded!")
    
    print()


def create_custom_exceptions():
    """Demonstrate custom exceptions."""
    print("=" * 60)
    print("CUSTOM EXCEPTIONS")
    print("=" * 60)
    
    # Define custom exceptions
    class ValidationError(Exception):
        """Base class for validation errors."""
        pass
    
    class TooShortError(ValidationError):
        """Raised when input is too short."""
        def __init__(self, value: str, min_length: int):
            self.value = value
            self.min_length = min_length
            super().__init__(f"'{value}' is too short (min {min_length} chars)")
    
    class InvalidFormatError(ValidationError):
        """Raised when format is invalid."""
        def __init__(self, value: str, expected_format: str):
            self.value = value
            self.expected_format = expected_format
            super().__init__(f"'{value}' doesn't match format '{expected_format}'")
    
    class AlreadyExistsError(ValidationError):
        """Raised when item already exists."""
        pass
    
    # Simulate user registration system
    existing_users = {"alice", "bob", "charlie"}
    
    def validate_username(username: str):
        """Validate username with custom exceptions."""
        print(f"\nValidating username: '{username}'")
        
        if len(username) < 3:
            raise TooShortError(username, 3)
        
        if not username.isalnum():
            raise InvalidFormatError(username, "alphanumeric")
        
        if username in existing_users:
            raise AlreadyExistsError(f"Username '{username}' already taken")
        
        return True
    
    # Test validation
    test_cases = [
        "ab",      # Too short
        "alice",   # Already exists
        "john_doe", # Contains underscore
        "ValidUser123", # Valid
        "a",       # Too short
        "bob",     # Already exists
        "newuser", # Valid
    ]
    
    for username in test_cases:
        try:
            if validate_username(username):
                print(f"  ✅ '{username}' is valid!")
                existing_users.add(username)
        except TooShortError as e:
            print(f"  ❌ {e}")
        except InvalidFormatError as e:
            print(f"  ❌ {e}")
        except AlreadyExistsError as e:
            print(f"  ❌ {e}")
    
    print()


def context_manager_example():
    """Demonstrate context managers for resource management."""
    print("=" * 60)
    print("CONTEXT MANAGERS AND WITH STATEMENT")
    print("=" * 60)
    
    # Built-in context managers
    print("1. Built-in context managers:")
    
    # File operations (automatically closes file)
    with open("example.txt", "w") as f:
        f.write("Hello from context manager!\n")
        print("  ✅ File written successfully")
    
    with open("example.txt", "r") as f:
        content = f.read()
        print(f"  ✅ File content: {content.strip()}")
    
    # Custom context manager class
    print("\n2. Custom context manager class:")
    
    class Timer:
        """Context manager for timing code blocks."""
        def __init__(self, name: str):
            self.name = name
            self.start_time = None
        
        def __enter__(self):
            import time
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            import time
            elapsed = time.time() - self.start_time
            print(f"  ⏱️  Timer '{self.name}': {elapsed:.4f} seconds")
            # Return False to propagate exceptions
            return False
    
    # Using custom context manager
    with Timer("List comprehension"):
        squares = [x**2 for x in range(10000)]
        print(f"  Generated {len(squares)} squares")
    
    print("\n3. Database connection simulation:")
    
    class DatabaseConnection:
        """Simulated database connection."""
        def __enter__(self):
            print("  Connecting to database...")
            # Simulate connection
            self.connected = True
            return self
        
        def execute(self, query: str):
            """Execute a query."""
            if not self.connected:
                raise ConnectionError("Not connected to database")
            print(f"  Executing: {query}")
            return [{"id": 1, "name": "example"}]
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            print("  Closing database connection...")
            self.connected = False
            # Handle exceptions if needed
            if exc_type:
                print(f"  ❌ Error in context: {exc_type.__name__}: {exc_val}")
            return True  # Suppress the exception
    
    with DatabaseConnection() as db:
        results = db.execute("SELECT * FROM users")
        print(f"  Results: {results}")
        
    print()


def advanced_patterns():
    """Demonstrate advanced error handling patterns."""
    print("=" * 60)
    print("ADVANCED ERROR HANDLING PATTERNS")
    print("=" * 60)
    
    # Pattern 1: Retry with exponential backoff
    print("1. Retry with exponential backoff:")
    
    import time
    
    def unreliable_operation(attempt: int):
        """Simulate an unreliable operation."""
        if attempt < 3:
            raise ConnectionError(f"Connection failed on attempt {attempt}")
        return "Success!"
    
    def retry_with_backoff(max_attempts: int = 5):
        """Retry operation with exponential backoff."""
        for attempt in range(1, max_attempts + 1):
            try:
                result = unreliable_operation(attempt)
                print(f"  ✅ Attempt {attempt}: {result}")
                return result
            except ConnectionError as e:
                if attempt == max_attempts:
                    print(f"  ❌ Failed after {max_attempts} attempts")
                    raise
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  ⏸️  Attempt {attempt} failed: {e}, waiting {wait_time}s")
                time.sleep(wait_time)
    
    try:
        retry_with_backoff()
    except ConnectionError:
        print("  Operation failed despite retries")
    
    # Pattern 2: Error aggregation
    print("\n2. Error aggregation:")
    
    def validate_data(data_list):
        """Validate multiple data items and aggregate errors."""
        errors = []
        valid_items = []
        
        for item in data_list:
            try:
                # Validate each item
                if not isinstance(item, (int, float)):
                    raise TypeError(f"Expected number, got {type(item).__name__}")
                if item < 0:
                    raise ValueError(f"Negative value: {item}")
                if item > 100:
                    raise ValueError(f"Too large: {item}")
                
                valid_items.append(item)
            except (TypeError, ValueError) as e:
                errors.append(f"Item '{item}': {e}")
        
        return valid_items, errors
    
    test_data = [42, -5, "hello", 150, 3.14, 99]
    valid, errors = validate_data(test_data)
    
    print(f"  Valid items: {valid}")
    print(f"  Errors ({len(errors)}):")
    for error in errors:
        print(f"    - {error}")
    
    # Pattern 3: Graceful degradation
    print("\n3. Graceful degradation:")
    
    def get_data_from_multiple_sources():
        """Try multiple data sources, use first successful one."""
        sources = [
            ("API", lambda: get_api_data()),
            ("Cache", lambda: get_cached_data()),
            ("Fallback", lambda: get_fallback_data()),
        ]
        
        for source_name, get_func in sources:
            try:
                data = get_func()
                print(f"  ✅ Got data from {source_name}: {data}")
                return data
            except Exception as e:
                print(f"  ⚠️  {source_name} failed: {e}")
                continue
        
        print("  ❌ All sources failed")
        return None
    
    def get_api_data():
        """Simulate API call."""
        raise ConnectionError("API timeout")
    
    def get_cached_data():
        """Simulate cache lookup."""
        return {"cached": "data"}
    
    def get_fallback_data():
        """Simulate fallback data."""
        return {"fallback": "default"}
    
    result = get_data_from_multiple_sources()
    print(f"  Final result: {result}")
    
    print()


def main():
    """Run all error handling examples."""
    print("\n" + "=" * 60)
    print("PYTHON ERROR HANDLING DEMONSTRATION")
    print("=" * 60)
    
    basic_error_handling()
    handle_mixed_errors()
    create_custom_exceptions()
    context_manager_example()
    advanced_patterns()
    
    print("\n" + "=" * 60)
    print("ERROR HANDLING BEST PRACTICES:")
    print("  1. Be specific about exceptions you catch")
    print("  2. Use custom exceptions for business logic errors")
    print("  3. Always clean up resources (use context managers)")
    print("  4. Log errors appropriately")
    print("  5. Provide helpful error messages")
    print("  6. Consider retry mechanisms for transient failures")
    print("  7. Validate input early and often")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Cleanup
try:
    import os
    os.remove("example_data.txt")
    os.remove("example.txt")
except FileNotFoundError:
    pass