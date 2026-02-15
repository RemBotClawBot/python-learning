"""
Object-Oriented Programming (OOP) Examples
-------------------------------------------
Demonstrates classes, inheritance, polymorphism, properties, and magic methods.
"""

class Animal:
    """Base class for all animals."""
    
    def __init__(self, name: str, species: str):
        self.name = name
        self.species = species
        self._age = 0  # Private attribute with underscore
    
    def speak(self) -> str:
        """All animals should implement this method."""
        return f"{self.name} the {self.species} makes a sound."
    
    def eat(self, food: str) -> str:
        """Generic eating method."""
        return f"{self.name} is eating {food}."
    
    @property
    def age(self) -> int:
        """Getter for age."""
        return self._age
    
    @age.setter
    def age(self, value: int):
        """Setter for age with validation."""
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value
    
    def __str__(self) -> str:
        """String representation of the animal."""
        return f"{self.name} ({self.species}), age: {self.age}"
    
    def __repr__(self) -> str:
        """Technical representation for debugging."""
        return f"Animal(name='{self.name}', species='{self.species}', age={self.age})"


class Dog(Animal):
    """Dog class inheriting from Animal."""
    
    def __init__(self, name: str, breed: str):
        super().__init__(name, "Dog")
        self.breed = breed
        self.tricks = []
    
    def speak(self) -> str:
        """Override speak method for dogs."""
        return f"{self.name} says: Woof! Woof!"
    
    def add_trick(self, trick: str):
        """Add a trick to the dog's repertoire."""
        self.tricks.append(trick)
        return f"{self.name} learned {trick}!"
    
    def __str__(self) -> str:
        """Enhanced string representation."""
        tricks_str = f", knows {len(self.tricks)} tricks" if self.tricks else ""
        return f"{self.name} ({self.breed} dog), age: {self.age}{tricks_str}"


class Cat(Animal):
    """Cat class inheriting from Animal."""
    
    def __init__(self, name: str, color: str):
        super().__init__(name, "Cat")
        self.color = color
        self._lives = 9
    
    def speak(self) -> str:
        """Override speak method for cats."""
        return f"{self.name} says: Meow!"
    
    @property
    def lives(self) -> int:
        """Getter for lives."""
        return self._lives
    
    def lose_life(self):
        """Cat loses a life."""
        if self._lives > 0:
            self._lives -= 1
            return f"{self.name} lost a life! {self._lives} lives remaining."
        return f"{self.name} has no lives left!"
    
    def __str__(self) -> str:
        """Enhanced string representation."""
        return f"{self.name} ({self.color} cat), age: {self.age}, lives: {self.lives}"


class BankAccount:
    """Simple bank account with encapsulation."""
    
    def __init__(self, owner: str, initial_balance: float = 0.0):
        self.owner = owner
        self._balance = initial_balance
        self.transaction_history = []
        self._add_transaction("Account opened", initial_balance)
    
    def deposit(self, amount: float) -> str:
        """Deposit money into account."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        self._balance += amount
        self._add_transaction("Deposit", amount)
        return f"Deposited ${amount:.2f}. New balance: ${self._balance:.2f}"
    
    def withdraw(self, amount: float) -> str:
        """Withdraw money from account."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        
        self._balance -= amount
        self._add_transaction("Withdrawal", -amount)
        return f"Withdrew ${amount:.2f}. New balance: ${self._balance:.2f}"
    
    @property
    def balance(self) -> float:
        """Getter for balance."""
        return self._balance
    
    def _add_transaction(self, description: str, amount: float):
        """Private method to add transaction to history."""
        self.transaction_history.append({
            "description": description,
            "amount": amount,
            "balance": self._balance,
            "timestamp": "2024-01-01 12:00:00"  # In real app, use datetime.now()
        })
    
    def get_statement(self) -> str:
        """Get formatted account statement."""
        statement = [f"Statement for {self.owner}'s account:"]
        statement.append(f"Current balance: ${self.balance:.2f}")
        statement.append("Transaction history:")
        
        for i, transaction in enumerate(self.transaction_history, 1):
            statement.append(f"  {i}. {transaction['description']}: "
                           f"${transaction['amount']:+.2f} "
                           f"(Balance: ${transaction['balance']:.2f})")
        
        return "\n".join(statement)
    
    def __str__(self) -> str:
        return f"BankAccount(owner='{self.owner}', balance=${self.balance:.2f})"


def demonstrate_polymorphism():
    """Demonstrate polymorphism with different animal types."""
    animals = [
        Dog("Rex", "Golden Retriever"),
        Cat("Whiskers", "Orange"),
        Animal("Generic", "Unknown Species")
    ]
    
    results = []
    for animal in animals:
        results.append(animal.speak())
        results.append(animal.eat("some food"))
    
    return results


def main():
    """Run all OOP examples."""
    print("=" * 60)
    print("OBJECT-ORIENTED PROGRAMMING EXAMPLES")
    print("=" * 60)
    
    # 1. Basic class example
    print("\n1. Basic Animal Class:")
    generic_animal = Animal("Spot", "Unknown")
    generic_animal.age = 3
    print(generic_animal)
    print(generic_animal.speak())
    print(generic_animal.eat("plants"))
    
    # 2. Inheritance example
    print("\n2. Dog Class (Inheritance):")
    my_dog = Dog("Buddy", "Labrador")
    my_dog.age = 2
    print(my_dog)
    print(my_dog.speak())
    print(my_dog.add_trick("sit"))
    print(my_dog.add_trick("roll over"))
    print(my_dog)
    
    # 3. Cat class with properties
    print("\n3. Cat Class (Properties):")
    my_cat = Cat("Mittens", "Gray")
    my_cat.age = 1
    print(my_cat)
    print(my_cat.speak())
    print(my_cat.lose_life())
    print(my_cat.lose_life())
    print(my_cat)
    
    # 4. Bank account with encapsulation
    print("\n4. Bank Account (Encapsulation):")
    account = BankAccount("John Doe", 1000.0)
    print(account)
    print(account.deposit(500))
    print(account.withdraw(200))
    print(account.withdraw(50))
    print(account.get_statement())
    
    # Try to withdraw too much
    try:
        account.withdraw(2000)
    except ValueError as e:
        print(f"Error: {e}")
    
    # 5. Polymorphism demonstration
    print("\n5. Polymorphism Example:")
    for result in demonstrate_polymorphism():
        print(f"  - {result}")
    
    # 6. Magic methods
    print("\n6. Magic Methods:")
    print(f"  str(repr): {repr(generic_animal)}")
    
    print("\n" + "=" * 60)
    print("OOP concepts demonstrated:")
    print("  - Classes and objects")
    print("  - Inheritance and method overriding")
    print("  - Encapsulation with private attributes")
    print("  - Properties (getters/setters)")
    print("  - Polymorphism")
    print("  - Magic methods (__str__, __repr__)")
    print("=" * 60)


if __name__ == "__main__":
    main()