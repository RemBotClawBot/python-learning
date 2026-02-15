"""
Type Hints and Static Typing Examples
---------------------------------------
Modern Python type annotations, typing module features,
and mypy/type checking demonstrations.
"""

from typing import (
    List, Dict, Tuple, Set, Optional, Union, Any,
    Callable, TypeVar, Generic, Iterator, Sequence,
    Literal, TypedDict, Protocol, runtime_checkable,
    NewType, overload
)
from dataclasses import dataclass, field
from enum import Enum
from typing import get_type_hints
import inspect
from datetime import datetime
import json
from decimal import Decimal
from pathlib import Path

# ============================================================================
# PART 1: BASIC TYPE ANNOTATIONS
# ============================================================================

def basic_type_examples():
    """Demonstrate basic type annotations."""
    
    # Variable annotations
    name: str = "Rem"
    age: int = 1
    temperature: float = 36.6
    is_learning: bool = True
    scores: List[int] = [95, 87, 92]
    
    # Function annotations
    def greet(person: str) -> str:
        return f"Hello, {person}!"
    
    def calculate_area(length: float, width: float) -> float:
        return length * width
    
    # Collections with types
    names: List[str] = ["Alice", "Bob", "Charlie"]
    ages: Dict[str, int] = {"Alice": 30, "Bob": 25}
    coordinates: Tuple[float, float, float] = (1.0, 2.0, 3.0)
    unique_ids: Set[int] = {1, 2, 3, 4, 5}
    
    # Optional types (can be None)
    middle_name: Optional[str] = None
    
    # Union types (multiple possible types)
    id_or_name: Union[int, str] = 123
    id_or_name = "Rem"
    
    print("✅ Basic type annotations examples")
    return locals()


# ============================================================================
# PART 2: FUNCTION SIGNATURES WITH TYPE HINTS
# ============================================================================

def process_data(
    data: List[Dict[str, Any]],
    filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
    max_items: int = 100
) -> List[Dict[str, Any]]:
    """
    Process data with type hints.
    
    Args:
        data: List of dictionaries to process
        filter_func: Optional function to filter items
        max_items: Maximum number of items to return
        
    Returns:
        Filtered and limited list of dictionaries
    """
    if filter_func:
        data = [item for item in data if filter_func(item)]
    
    return data[:max_items]


@overload
def parse_value(value: str) -> str: ...
@overload
def parse_value(value: int) -> int: ...
@overload
def parse_value(value: float) -> float: ...

def parse_value(value: Union[str, int, float]) -> Union[str, int, float]:
    """
    Overloaded function with type hints.
    
    This demonstrates how to use @overload decorator
    to provide better type hints for different input types.
    """
    if isinstance(value, str):
        return value.upper()
    elif isinstance(value, int):
        return value * 2
    elif isinstance(value, float):
        return round(value, 2)
    else:
        raise TypeError(f"Unsupported type: {type(value)}")


# ============================================================================
# PART 3: CUSTOM TYPES AND TYPE VARIABLES
# ============================================================================

# Type variables for generic functions
T = TypeVar('T')
U = TypeVar('U')
Number = TypeVar('Number', int, float, Decimal)


def swap(a: T, b: T) -> Tuple[T, T]:
    """Swap two values of the same type."""
    return b, a


def merge_dicts(dict1: Dict[T, U], dict2: Dict[T, U]) -> Dict[T, U]:
    """Merge two dictionaries."""
    return {**dict1, **dict2}


def sum_numbers(numbers: List[Number]) -> Number:
    """Sum a list of numbers (int, float, or Decimal)."""
    return sum(numbers)  # type: ignore


# Custom type definitions
UserId = NewType('UserId', int)
Email = NewType('Email', str)


def create_user(user_id: UserId, email: Email) -> Dict[str, Any]:
    """Create user with strongly typed IDs."""
    return {"id": user_id, "email": email}


# TypedDict for dictionary schemas
class User(TypedDict, total=False):
    """Type definition for user dictionaries."""
    id: int
    name: str
    email: str
    age: Optional[int]
    is_active: bool


def validate_user(user: User) -> bool:
    """Validate user dictionary against schema."""
    required = {"id", "name", "email"}
    return all(key in user for key in required)


# ============================================================================
# PART 4: DATACLASSES AND ENUMS WITH TYPE HINTS
# ============================================================================

class Status(Enum):
    """Status enumeration with type hints."""
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DELETED = "deleted"


@dataclass
class Product:
    """Product data class with type hints."""
    id: int
    name: str
    price: float
    category: str
    in_stock: bool = True
    tags: List[str] = field(default_factory=list)
    
    def formatted_price(self) -> str:
        return f"${self.price:.2f}"
    
    @property
    def is_expensive(self) -> bool:
        return self.price > 100.0


@dataclass
class Order:
    """Order data class with type hints."""
    order_id: int
    customer_id: int
    products: List[Product]
    status: Status = Status.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    
    def total_price(self) -> float:
        return sum(p.price for p in self.products)
    
    def mark_shipped(self) -> None:
        self.status = Status.ACTIVE


# ============================================================================
# PART 5: PROTOCOLS AND STRUCTURAL TYPING
# ============================================================================

@runtime_checkable
class Printable(Protocol):
    """Protocol for printable objects."""
    
    def print(self) -> str:
        ...


@dataclass
class Document:
    """Document implementing Printable protocol."""
    title: str
    content: str
    
    def print(self) -> str:
        return f"Document: {self.title}\n{self.content}"


@dataclass
class Report:
    """Report implementing Printable protocol."""
    name: str
    data: Dict[str, Any]
    
    def print(self) -> str:
        return f"Report: {self.name}\n{json.dumps(self.data, indent=2)}"


def print_items(items: Sequence[Printable]) -> List[str]:
    """Print a sequence of printable items."""
    return [item.print() for item in items]


# ============================================================================
# PART 6: TYPE CHECKING AT RUNTIME
# ============================================================================

def dynamic_type_checking():
    """Demonstrate runtime type checking."""
    
    # Get type hints programmatically
    hints = get_type_hints(process_data)
    print("Type hints for process_data:")
    for param, type_hint in hints.items():
        print(f"  {param}: {type_hint}")
    
    # Inspect function signatures
    sig = inspect.signature(process_data)
    print("\nSignature of process_data:")
    print(f"  {sig}")
    
    # Check types at runtime
    def add_numbers(a: int, b: int) -> int:
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("Arguments must be integers")
        return a + b
    
    try:
        result = add_numbers(5, 3)
        print(f"\nRuntime type check passed: {result}")
        
        # This will raise TypeError
        result = add_numbers("5", 3)
    except TypeError as e:
        print(f"\nRuntime type check caught error: {e}")
    
    return hints


# ============================================================================
# PART 7: PRACTICAL EXAMPLES WITH TYPE HINTS
# ============================================================================

def read_csv_file(filepath: Path) -> List[Dict[str, Any]]:
    """
    Read CSV file with type hints.
    
    Note: In real code, you'd use pandas or csv module.
    """
    # Simulate reading CSV
    return [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
        {"id": 3, "name": "Charlie", "age": 35}
    ]


def filter_users(
    users: List[Dict[str, Any]],
    min_age: Optional[int] = None,
    name_contains: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Filter users with optional criteria."""
    filtered = users
    
    if min_age is not None:
        filtered = [u for u in filtered if u.get("age", 0) >= min_age]
    
    if name_contains is not None:
        filtered = [
            u for u in filtered 
            if name_contains.lower() in u.get("name", "").lower()
        ]
    
    return filtered


def calculate_stats(
    numbers: List[Union[int, float]]
) -> Dict[str, Union[int, float]]:
    """Calculate statistics with type hints."""
    if not numbers:
        return {"count": 0, "sum": 0, "average": 0.0}
    
    count = len(numbers)
    total = sum(numbers)
    average = total / count
    maximum = max(numbers)
    minimum = min(numbers)
    
    return {
        "count": count,
        "sum": total,
        "average": average,
        "max": maximum,
        "min": minimum
    }


# ============================================================================
# PART 8: USING MYPY FOR STATIC TYPE CHECKING
# ============================================================================

def demonstrate_type_errors():
    """
    Functions that would cause mypy type errors.
    
    Uncomment any of these to see mypy errors:
    $ python -m mypy type_hints_examples.py
    """
    
    # 1. Type mismatch
    # name: str = 42  # mypy error: Incompatible types in assignment
    
    # 2. Wrong return type
    # def get_name() -> str:
    #     return 123  # mypy error: Incompatible return value type
    
    # 3. Missing argument
    # def greet(name: str) -> str:
    #     return f"Hello {name}"
    # greet()  # mypy error: Missing positional argument "name"
    
    # 4. Wrong argument type
    # greet(42)  # mypy error: Argument 1 has incompatible type
    
    # 5. Accessing non-existent attribute
    # user = {"name": "Rem"}
    # age = user["age"]  # mypy might not catch this without TypedDict
    
    print("⚠️  This function demonstrates potential type errors.")
    print("   Run 'python -m mypy type_hints_examples.py' to check.")


# ============================================================================
# PART 9: TYPE HINTS BEST PRACTICES
# ============================================================================

def type_hints_best_practices():
    """Demonstrate best practices for type hints."""
    
    # 1. Use Optional instead of Union[Type, None]
    # Good:
    def get_user(id: int) -> Optional[Dict[str, Any]]:
        return None
    
    # 2. Be specific with collections
    # Good:
    users: List[Dict[str, Union[str, int]]]
    # Less good:
    # users: list
    
    # 3. Use TypedDict for dictionary schemas
    # Good when you know the structure
    
    # 4. Use NewType for type safety
    # Good for IDs, emails, etc.
    
    # 5. Add type hints to class methods
    class Calculator:
        def add(self, a: int, b: int) -> int:
            return a + b
    
    # 6. Use @overload for complex signatures
    
    # 7. Document with docstrings that reference types
    
    return "Best practices documented"


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Run all type hints examples."""
    print("🔍 Type Hints and Static Typing Examples")
    print("="*60)
    
    try:
        # Part 1: Basic type annotations
        print("\n1️⃣  Basic Type Annotations")
        locals_scope = basic_type_examples()
        print("   Variables created:", list(locals_scope.keys()))
        
        # Part 2: Function signatures
        print("\n2️⃣  Function Signatures")
        data = [{"id": i, "value": i*10} for i in range(5)]
        result = process_data(data, max_items=3)
        print(f"   Processed {len(result)} items")
        
        # Part 3: Custom types
        print("\n3️⃣  Custom Types")
        user_id = UserId(123)
        email = Email("rem@company.com")
        user = create_user(user_id, email)
        print(f"   Created user: {user}")
        
        # Part 4: Dataclasses and Enums
        print("\n4️⃣  Dataclasses and Enums")
        product = Product(id=1, name="Laptop", price=999.99, category="Electronics")
        print(f"   Product: {product.name} - {product.formatted_price()}")
        print(f"   Expensive: {product.is_expensive}")
        
        # Part 5: Protocols
        print("\n5️⃣  Protocols (Structural Typing)")
        doc = Document(title="Report", content="Some content")
        report = Report(name="Sales", data={"q1": 1000, "q2": 1500})
        print(f"   Document: {doc.print()[:50]}...")
        print(f"   Report: {report.print()[:50]}...")
        
        # Part 6: Runtime type checking
        print("\n6️⃣  Runtime Type Checking")
        hints = dynamic_type_checking()
        print(f"   Found {len(hints)} type hints")
        
        # Part 7: Practical examples
        print("\n7️⃣  Practical Examples")
        users = read_csv_file(Path("users.csv"))
        filtered = filter_users(users, min_age=30)
        print(f"   Found {len(filtered)} users age 30+")
        
        stats = calculate_stats([10, 20, 30, 40, 50])
        print(f"   Stats: {stats}")
        
        # Part 8: Best practices
        print("\n8️⃣  Type Hints Best Practices")
        best_practices = type_hints_best_practices()
        print(f"   {best_practices}")
        
        print("\n" + "="*60)
        print("🎉 Type hints examples completed!")
        print("\n📚 Key Benefits of Type Hints:")
        print("   1. Better documentation and code readability")
        print("   2. Catch type-related bugs early")
        print("   3. Enable better IDE support (autocomplete, refactoring)")
        print("   4. Improve code maintainability")
        print("   5. Facilitate team collaboration")
        
        print("\n🔧 Tools for Type Checking:")
        print("   - mypy: Static type checker")
        print("   - pyright: Microsoft's type checker")
        print("   - pyre: Facebook's type checker")
        print("   - pytype: Google's type checker")
        
        print("\n💡 Try running: python -m mypy type_hints_examples.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()