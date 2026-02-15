# lists_loops.py
"""
Comprehensive examples of Python lists, loops, and list comprehensions.
"""

def demonstrate_lists():
    """Basic list operations."""
    print("=" * 60)
    print("LIST OPERATIONS")
    print("=" * 60)
    
    # Creating lists
    fruits = ["apple", "banana", "cherry", "date"]
    numbers = [1, 2, 3, 4, 5]
    mixed_list = ["hello", 42, 3.14, True]
    empty_list = []
    
    print("\n1. Creating Lists:")
    print(f"  fruits = {fruits}")
    print(f"  numbers = {numbers}")
    print(f"  mixed_list = {mixed_list}")
    print(f"  empty_list = {empty_list}")
    
    # Accessing elements
    print("\n2. Accessing Elements:")
    print(f"  fruits[0] = {fruits[0]}")
    print(f"  fruits[-1] = {fruits[-1]}")  # Last element
    print(f"  fruits[1:3] = {fruits[1:3]}")  # Slicing
    print(f"  fruits[:2] = {fruits[:2]}")     # First two
    print(f"  fruits[2:] = {fruits[2:]}")     # From index 2
    
    # List methods
    print("\n3. List Methods:")
    fruits.append("elderberry")
    print(f"  After append('elderberry'): {fruits}")
    
    fruits.insert(2, "blueberry")
    print(f"  After insert(2, 'blueberry'): {fruits}")
    
    removed = fruits.pop()
    print(f"  After pop(): removed '{removed}', list: {fruits}")
    
    fruits.remove("banana")
    print(f"  After remove('banana'): {fruits}")
    
    fruits.sort()
    print(f"  After sort(): {fruits}")
    
    fruits.reverse()
    print(f"  After reverse(): {fruits}")
    
    print(f"  Length: {len(fruits)}")
    print(f"  Index of 'cherry': {fruits.index('cherry')}")
    print(f"  Count of 'apple': {fruits.count('apple')}")
    
    # List concatenation
    print("\n4. List Concatenation:")
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    combined = list1 + list2
    print(f"  {list1} + {list2} = {combined}")
    
    # List multiplication
    repeated = ["ha"] * 3
    print(f"  ['ha'] * 3 = {repeated}")


def demonstrate_loops():
    """Different types of loops in Python."""
    print("\n" + "=" * 60)
    print("LOOPS")
    print("=" * 60)
    
    numbers = [10, 20, 30, 40, 50]
    fruits = ["apple", "banana", "cherry", "date"]
    
    print("\n1. For Loop (basic):")
    print("  Iterating through fruits:")
    for fruit in fruits:
        print(f"    I like {fruit}s")
    
    print("\n2. For Loop with enumerate():")
    print("  Getting index and value:")
    for index, fruit in enumerate(fruits):
        print(f"    fruits[{index}] = {fruit}")
    
    print("\n3. For Loop with range():")
    print("  Using range to iterate numbers:")
    for i in range(5):
        print(f"    i = {i}")
    
    print("\n4. For Loop with range(start, stop, step):")
    print("  Counting by twos:")
    for i in range(0, 10, 2):
        print(f"    i = {i}")
    
    print("\n5. While Loop:")
    print("  Countdown from 5:")
    count = 5
    while count > 0:
        print(f"    {count}...")
        count -= 1
    print("    Liftoff!")
    
    print("\n6. Loop Control Statements:")
    print("  break and continue examples:")
    for i in range(10):
        if i == 2:
            continue  # Skip 2
        if i == 7:
            break     # Stop at 7
        print(f"    i = {i}")
    
    print("\n7. Nested Loops:")
    print("  Multiplication table (1-3):")
    for i in range(1, 4):
        for j in range(1, 4):
            print(f"    {i} × {j} = {i * j}")
        print()  # Blank line between tables


def demonstrate_list_comprehensions():
    """List comprehensions and generator expressions."""
    print("\n" + "=" * 60)
    print("LIST COMPREHENSIONS")
    print("=" * 60)
    
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print("\n1. Basic List Comprehension:")
    squares = [x**2 for x in range(1, 6)]
    print(f"  [x**2 for x in range(1, 6)] = {squares}")
    
    print("\n2. List Comprehension with Condition:")
    evens = [x for x in numbers if x % 2 == 0]
    print(f"  Even numbers: {evens}")
    
    odds = [x for x in numbers if x % 2 != 0]
    print(f"  Odd numbers: {odds}")
    
    print("\n3. Transforming Elements:")
    words = ["hello", "world", "python", "learning"]
    uppercase_words = [word.upper() for word in words]
    print(f"  Uppercase words: {uppercase_words}")
    
    word_lengths = [len(word) for word in words]
    print(f"  Word lengths: {word_lengths}")
    
    print("\n4. Nested List Comprehension:")
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flattened = [num for row in matrix for num in row]
    print(f"  Flattened matrix: {flattened}")
    
    print("\n5. Dictionary Comprehension:")
    word_dict = {word: len(word) for word in words}
    print(f"  Dictionary {{word: length}}: {word_dict}")
    
    print("\n6. Set Comprehension:")
    numbers_with_duplicates = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    unique_numbers = {x for x in numbers_with_duplicates}
    print(f"  Unique numbers: {unique_numbers}")
    
    print("\n7. Generator Expression (lazy evaluation):")
    big_numbers = (x for x in range(1000000) if x % 100000 == 0)
    print("  First few from generator of 1 million numbers:")
    for i, num in enumerate(big_numbers):
        if i < 5:
            print(f"    {num}")
        else:
            break


def practical_examples():
    """Practical examples using lists and loops."""
    print("\n" + "=" * 60)
    print("PRACTICAL EXAMPLES")
    print("=" * 60)
    
    print("\n1. Shopping Cart:")
    cart = [
        {"item": "Apple", "price": 0.50, "quantity": 3},
        {"item": "Banana", "price": 0.30, "quantity": 5},
        {"item": "Milk", "price": 2.99, "quantity": 1},
        {"item": "Bread", "price": 1.99, "quantity": 2}
    ]
    
    print("  Shopping Cart:")
    total = 0
    for product in cart:
        item_total = product["price"] * product["quantity"]
        total += item_total
        print(f"    {product['item']}: {product['quantity']} × ${product['price']:.2f} = ${item_total:.2f}")
    print(f"  Total: ${total:.2f}")
    
    print("\n2. Student Grades:")
    students = [
        {"name": "Alice", "grades": [85, 90, 78]},
        {"name": "Bob", "grades": [92, 88, 94]},
        {"name": "Charlie", "grades": [76, 81, 79]},
        {"name": "Diana", "grades": [88, 92, 85]}
    ]
    
    print("  Student Averages:")
    for student in students:
        average = sum(student["grades"]) / len(student["grades"])
        print(f"    {student['name']}: {student['grades']} = {average:.1f} average")
    
    print("\n3. Matrix Operations:")
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]
    
    print("  Matrix Addition:")
    result = [[matrix_a[i][j] + matrix_b[i][j] for j in range(len(matrix_a[0]))] 
              for i in range(len(matrix_a))]
    
    for i in range(len(matrix_a)):
        print(f"    {matrix_a[i]} + {matrix_b[i]} = {result[i]}")
    
    print("\n4. Finding Maximum and Minimum:")
    temperatures = [22.5, 24.1, 19.8, 25.3, 21.7, 18.9, 23.4]
    print(f"  Temperatures: {temperatures}")
    print(f"  Max temperature: {max(temperatures):.1f}°C")
    print(f"  Min temperature: {min(temperatures):.1f}°C")
    print(f"  Average temperature: {sum(temperatures)/len(temperatures):.1f}°C")
    
    print("\n5. Filtering Data:")
    inventory = [
        {"product": "Widget", "stock": 45, "price": 12.99},
        {"product": "Gadget", "stock": 12, "price": 24.50},
        {"product": "Thingy", "stock": 0, "price": 8.75},
        {"product": "Doodad", "stock": 23, "price": 15.25},
        {"product": "Whatchamacallit", "stock": 5, "price": 32.99}
    ]
    
    low_stock = [item for item in inventory if item["stock"] < 10]
    print("  Low stock items (less than 10):")
    for item in low_stock:
        print(f"    {item['product']}: {item['stock']} in stock")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("PYTHON LISTS AND LOOPS TUTORIAL")
    print("=" * 60)
    
    demonstrate_lists()
    demonstrate_loops()
    demonstrate_list_comprehensions()
    practical_examples()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Concepts Covered:
1. Lists: Creation, indexing, slicing, methods
2. Loops: for, while, enumerate(), range()
3. List Comprehensions: Transformations, filtering
4. Practical Applications: Shopping cart, grades, matrices

Practice Exercises:
1. Create a list of your favorite movies and print each one
2. Calculate the sum of all even numbers from 1 to 100
3. Use list comprehension to create a list of squares of odd numbers 1-20
4. Write a function that finds the most common element in a list
5. Create a multiplication table for numbers 1-10 using nested loops
    """)
    
    print("\n✅ All examples completed successfully!")


if __name__ == "__main__":
    main()