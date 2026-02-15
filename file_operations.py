# file_operations.py
import os
import json

def write_text_file(filename, content):
    """Write text content to a file."""
    with open(filename, "w") as file:
        file.write(content)
    print(f"✓ Written to {filename}")
    return True

def read_text_file(filename):
    """Read text content from a file."""
    try:
        with open(filename, "r") as file:
            content = file.read()
        print(f"✓ Read from {filename}")
        return content
    except FileNotFoundError:
        print(f"✗ File {filename} not found")
        return None

def append_to_file(filename, content):
    """Append content to a file."""
    with open(filename, "a") as file:
        file.write(content)
    print(f"✓ Appended to {filename}")
    return True

def write_json_file(filename, data):
    """Write data as JSON to a file."""
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)
    print(f"✓ JSON written to {filename}")
    return True

def read_json_file(filename):
    """Read JSON data from a file."""
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        print(f"✓ JSON read from {filename}")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"✗ Error reading {filename}: {e}")
        return None

def list_files_in_directory(directory="."):
    """List files in a directory."""
    try:
        files = os.listdir(directory)
        print(f"Files in {directory}:")
        for file in files:
            filepath = os.path.join(directory, file)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"  📄 {file} ({size} bytes)")
            else:
                print(f"  📁 {file}/")
        return files
    except FileNotFoundError:
        print(f"✗ Directory {directory} not found")
        return []

# Example usage
if __name__ == "__main__":
    print("File Operations Examples")
    print("-" * 40)
    
    # 1. Write and read text file
    print("\n1. Text File Operations:")
    write_text_file("example.txt", "Hello from Python!\nThis is a text file.\n")
    content = read_text_file("example.txt")
    print(f"Content: \n{content}")
    
    # 2. Append to file
    print("\n2. Appending to File:")
    append_to_file("example.txt", "Appended line!\n")
    updated_content = read_text_file("example.txt")
    print(f"Updated content: \n{updated_content}")
    
    # 3. JSON File Operations
    print("\n3. JSON File Operations:")
    user_data = {
        "name": "Rem",
        "skills": ["Python", "Git", "Automation"],
        "age_days": 1,
        "is_learning": True
    }
    write_json_file("user_data.json", user_data)
    loaded_data = read_json_file("user_data.json")
    print(f"Loaded data: {loaded_data}")
    
    # 4. List files
    print("\n4. Directory Listing:")
    list_files_in_directory()
    
    # 5. File existence check
    print("\n5. File Existence Check:")
    files_to_check = ["example.txt", "user_data.json", "nonexistent.txt"]
    for file in files_to_check:
        exists = os.path.exists(file)
        print(f"  {file}: {'Exists' if exists else 'Does not exist'}")
    
    # Cleanup (optional)
    print("\n6. Cleanup (uncomment to remove files):")
    # os.remove("example.txt")
    # os.remove("user_data.json")
    # print("Temporary files removed")
    
    print("\n✅ File operations completed successfully!")