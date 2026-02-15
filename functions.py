# functions.py

def greet(name):
    """Return a greeting message."""
    return f"Hello, {name}!"

def calculate_area(length, width):
    """Calculate area of a rectangle."""
    return length * width

def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def factorial(n):
    """Calculate factorial of n."""
    if n == 0:
        return 1
    return n * factorial(n - 1)

def is_even(number):
    """Check if a number is even."""
    return number % 2 == 0

# Using functions
print("Function Examples:")
print("-" * 40)

# Greeting
message = greet("Python Learner")
print(f"greet('Python Learner'): {message}")

# Area calculation
area = calculate_area(5, 3)
print(f"calculate_area(5, 3): {area} square units")

# Temperature conversion
c_temp = 25
f_temp = celsius_to_fahrenheit(c_temp)
print(f"celsius_to_fahrenheit(25): {c_temp}°C = {f_temp:.1f}°F")

# Factorial
num = 5
fact = factorial(num)
print(f"factorial(5): {num}! = {fact}")

# Even/Odd check
test_nums = [2, 7, 10, 15]
for num in test_nums:
    result = "even" if is_even(num) else "odd"
    print(f"is_even({num}): {num} is {result}")

# Function composition
def calculate_volume(length, width, height):
    """Calculate volume using area function."""
    base_area = calculate_area(length, width)
    return base_area * height

volume = calculate_volume(3, 4, 5)
print(f"\ncalculate_volume(3, 4, 5): Volume = {volume} cubic units")

# Default parameters
def create_user(username, is_admin=False, is_active=True):
    """Create user with default parameters."""
    return {
        "username": username,
        "admin": is_admin,
        "active": is_active
    }

user1 = create_user("alice")
user2 = create_user("bob", is_admin=True)
print(f"\ncreate_user('alice'): {user1}")
print(f"create_user('bob', is_admin=True): {user2}")