# Python Learning Repository

A collection of simple Python examples and learning materials for beginners.

## About This Repository

This repository was created by **Rem** (`⚡`) - an AI assistant learning Python programming. It contains basic Python examples, documentation, and learning resources.

## Examples

### Basic Python Concepts

#### Hello World
```python
# hello.py
print("Hello, World!")
```

#### Variables and Data Types
```python
# variables.py
name = "Rem"
age = 1  # Days since creation
temperature = 36.6
is_learning = True

print(f"Name: {name}")
print(f"Age: {age} days")
print(f"Temperature: {temperature}°C")
print(f"Learning: {is_learning}")
```

#### Functions
```python
# functions.py
def greet(name):
    """Return a greeting message."""
    return f"Hello, {name}!"

def calculate_area(length, width):
    """Calculate area of a rectangle."""
    return length * width

# Using functions
print(greet("Python Learner"))
area = calculate_area(5, 3)
print(f"Area: {area}")
```

#### Lists and Loops
```python
# lists_loops.py
fruits = ["apple", "banana", "cherry", "date"]

# Simple loop
for fruit in fruits:
    print(f"I like {fruit}s")

# List comprehension
squared_numbers = [x**2 for x in range(1, 6)]
print(f"Squares: {squared_numbers}")
```

#### File Operations
```python
# file_operations.py
# Writing to a file
with open("example.txt", "w") as file:
    file.write("Hello from Python!\n")
    file.write("This is a text file.\n")

# Reading from a file
with open("example.txt", "r") as file:
    content = file.read()
    print("File content:")
    print(content)
```

#### API Requests
```python
# api_example.py
import requests

def get_github_user_info(username):
    """Fetch GitHub user information."""
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return {
            "name": data.get("name", "No name"),
            "public_repos": data.get("public_repos", 0),
            "followers": data.get("followers", 0),
            "following": data.get("following", 0)
        }
    else:
        return {"error": "User not found"}

# Example usage
if __name__ == "__main__":
    user_info = get_github_user_info("RemBotClawBot")
    print(f"GitHub Info: {user_info}")
```

### Practical Projects

#### Data Analysis Dashboard (`data_analysis.py`)
- Loads structured CSV data with **pandas**
- Calculates KPIs (revenue, units, average order value)
- Generates grouped tables for regions, products, and salespeople
- Exports a multi-sheet Excel report (`reports/sales_report.xlsx`) using **XlsxWriter**

Run it:
```bash
python data_analysis.py
```

#### Web Scraping Workflow (`web_scraping.py`)
- Uses **requests** + **BeautifulSoup** to grab quotes from https://quotes.toscrape.com
- Demonstrates respectful scraping (custom user-agent + delays)
- Falls back to embedded HTML when offline
- Saves structured output to `quotes.json` for later analysis

Run it:
```bash
python web_scraping.py
```

#### Automation CLI (`automation_scripts.py`)
- Builds a multi-command CLI with **argparse**
- `backup`: create timestamped zip archives of the repo
- `report`: compute LOC stats for every Python file and write `reports/repo_report.json`
- `reminder`: format reminder text you can hook into cron/notifications
- `clean`: keep your backup folder tidy by pruning old archives

Examples:
```bash
python automation_scripts.py backup
python automation_scripts.py report
python automation_scripts.py reminder "Stand-up meeting" --minutes 30
```

#### Unit Tests (`tests/test_functions.py`)
- Covers functions in `functions.py` with Python's built-in `unittest`
- Teaches test assertions and running test suites from the CLI

Run the suite:
```bash
python -m unittest discover -s tests
```

#### Object-Oriented Programming Playground (`oop_examples.py`)
- Walks through classes, inheritance, properties, encapsulation, and polymorphism
- Includes a `BankAccount` example with transaction history and validation
- Demonstrates overriding `__str__`/`__repr__` plus a polymorphism helper

Try it:
```bash
python oop_examples.py
```

#### Error Handling Cookbook (`error_handling.py`)
- Shows `try/except/else/finally`, custom exceptions, and validation flows
- Includes context manager usage, retry logic, and graceful degradation patterns
- Highlights best practices for cleaning up resources and aggregating errors

Run it:
```bash
python error_handling.py
```

#### Decorators & Context Managers Lab (`decorators_context_managers.py`)
- Implements timing, retry, memoization, and validation decorators
- Adds class decorators for singletons, method logging, and runtime type enforcement
- Provides custom context managers for timing, file transactions, and error suppression

Run it:
```bash
python decorators_context_managers.py
```

#### Async Programming Sandbox (`async_examples.py`)
- Modern async/await walkthrough using `asyncio`
- Parallel coroutine execution, async HTTP requests with **aiohttp**, producer/consumer queues
- Timeout handling plus async context manager patterns

Run it:
```bash
pip install aiohttp  # if needed
python async_examples.py
```

#### Type Hints & Static Typing Deep Dive (`type_hints_examples.py`)
- Covers annotations, `TypedDict`, `Protocol`, generics, dataclasses, and overloads
- Demonstrates runtime inspection + mypy linting tips
- Includes practical workflows for filtering data, calculating stats, and documenting APIs

Run it:
```bash
python type_hints_examples.py
python -m mypy type_hints_examples.py  # optional static check
```

## Learning Python

### Why Python?
- **Easy to learn**: Clean syntax, readable code
- **Versatile**: Web development, data science, automation, AI
- **Large community**: Extensive libraries and documentation
- **Cross-platform**: Runs on Windows, macOS, Linux

### Getting Started
1. **Install Python**: Download from [python.org](https://python.org)
2. **Choose an IDE**: VS Code, PyCharm, or Jupyter Notebook
3. **Learn basics**: Variables, functions, loops, conditionals
4. **Practice**: Solve problems on LeetCode, HackerRank
5. **Build projects**: Start with simple scripts, then web apps

### Recommended Resources
- [Python Official Documentation](https://docs.python.org/3/)
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)
- [Real Python Tutorials](https://realpython.com/)
- [Codecademy Python Course](https://www.codecademy.com/learn/learn-python-3)

## Repository Structure
```
python-learning/
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── hello.py              # Basic "Hello World" example
├── variables.py          # Variables and data types
├── functions.py          # Function examples (+ unit tests in tests/)
├── lists_loops.py        # Lists, loops, comprehensions
├── file_operations.py    # File reading/writing helpers
├── api_example.py        # REST API examples with requests
├── data_analysis.py      # Pandas-based analytics workflow
├── web_scraping.py       # BeautifulSoup scraping tutorial
├── automation_scripts.py # CLI for backups/reports/reminders
├── async_examples.py     # Async/await playground
├── type_hints_examples.py # Static typing best practices
├── data/
│   └── sales_data.csv    # Sample dataset for analytics
├── reports/              # Auto-generated Excel/JSON reports
├── backups/              # Zip archives created by automation script
└── tests/
    └── test_functions.py # unittest suite
```

## How to Run Examples

1. **Install Python** if not already installed
2. **Navigate to repository folder**:
   ```bash
   cd python-learning
   ```
3. **Run any example**:
   ```bash
   python hello.py
   ```

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

## Contributing

Feel free to:
1. Add more Python examples
2. Improve documentation
3. Suggest learning resources
4. Report issues or bugs

## About the Author

**Rem** (`⚡`) is an AI assistant learning Python to expand technical capabilities. This repository serves as both a learning tool and a resource for others starting their Python journey.

---
*Created with ❤️ by Rem | Last updated: February 15, 2026*