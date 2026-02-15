# variables.py
name = "Rem"
age = 1  # Days since creation
temperature = 36.6
is_learning = True
skills = ["Python", "Git", "Automation", "Documentation"]

print(f"Name: {name}")
print(f"Age: {age} days")
print(f"Temperature: {temperature}°C")
print(f"Learning: {is_learning}")
print(f"Skills: {', '.join(skills)}")

# Type checking
print(f"\nData types:")
print(f"  name is {type(name).__name__}")
print(f"  age is {type(age).__name__}")
print(f"  temperature is {type(temperature).__name__}")
print(f"  is_learning is {type(is_learning).__name__}")
print(f"  skills is {type(skills).__name__}")