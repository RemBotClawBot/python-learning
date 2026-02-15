# api_example.py
"""
Comprehensive API examples using Python's requests library.
Covers REST APIs, JSON handling, error handling, and practical use cases.
"""

import requests
import json
import time
from datetime import datetime


def basic_api_request():
    """Demonstrate basic API request and response handling."""
    print("=" * 60)
    print("BASIC API REQUESTS")
    print("=" * 60)
    
    # GitHub Users API
    print("\n1. GitHub User Information:")
    username = "octocat"  # GitHub's mascot
    url = f"https://api.github.com/users/{username}"
    
    try:
        response = requests.get(url)
        
        print(f"  Request URL: {url}")
        print(f"  Status Code: {response.status_code}")
        print(f"  Response Time: {response.elapsed.total_seconds():.3f}s")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  User: {data.get('name', 'No name')}")
            print(f"  Username: {data.get('login')}")
            print(f"  Public Repos: {data.get('public_repos', 0)}")
            print(f"  Followers: {data.get('followers', 0)}")
            print(f"  Following: {data.get('following', 0)}")
            print(f"  Created: {data.get('created_at', 'Unknown')}")
        else:
            print(f"  Error: {response.status_code} - {response.reason}")
            
    except requests.exceptions.RequestException as e:
        print(f"  Request failed: {e}")
    
    # JSONPlaceholder API (fake REST API)
    print("\n2. JSONPlaceholder - Posts API:")
    posts_url = "https://jsonplaceholder.typicode.com/posts/1"
    
    try:
        response = requests.get(posts_url)
        if response.status_code == 200:
            post = response.json()
            print(f"  Post ID: {post.get('id')}")
            print(f"  Title: {post.get('title')}")
            print(f"  Body: {post.get('body')[:50]}...")
        else:
            print(f"  Error: {response.status_code}")
    except Exception as e:
        print(f"  Error: {e}")


def api_with_parameters():
    """Demonstrate API requests with query parameters."""
    print("\n" + "=" * 60)
    print("API REQUESTS WITH PARAMETERS")
    print("=" * 60)
    
    # OpenWeatherMap API (using free tier example)
    print("\n1. Weather API Example (Mock):")
    
    # Note: In a real scenario, you'd need an API key
    # For demonstration, we'll use a mock response
    print("  This would typically require an API key")
    print("  Example endpoint: https://api.openweathermap.org/data/2.5/weather")
    print("  Parameters: q=London&appid=YOUR_API_KEY&units=metric")
    
    # Instead, let's use a public no-auth API
    print("\n2. Public User API with Parameters:")
    users_url = "https://jsonplaceholder.typicode.com/users"
    
    params = {
        "_limit": 3,      # Limit to 3 users
        "_sort": "name",   # Sort by name
        "_order": "asc"    # Ascending order
    }
    
    try:
        response = requests.get(users_url, params=params)
        print(f"  Request URL: {response.url}")
        
        if response.status_code == 200:
            users = response.json()
            print(f"  Found {len(users)} users:")
            for user in users:
                print(f"    - {user.get('name')} ({user.get('email')})")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n3. GitHub Search API:")
    search_url = "https://api.github.com/search/repositories"
    search_params = {
        "q": "python",
        "sort": "stars",
        "order": "desc",
        "per_page": 3
    }
    
    try:
        response = requests.get(search_url, params=search_params)
        if response.status_code == 200:
            data = response.json()
            print(f"  Found {data.get('total_count', 0)} Python repositories")
            print(f"  Top 3 by stars:")
            
            for repo in data.get('items', [])[:3]:
                name = repo.get('name')
                stars = repo.get('stargazers_count', 0)
                description = repo.get('description', 'No description')
                print(f"    ⭐ {name} ({stars:,} stars)")
                print(f"      {description[:60]}...")
        else:
            print(f"  Error: {response.status_code} - {response.reason}")
    except Exception as e:
        print(f"  Error: {e}")


def working_with_json():
    """Demonstrate JSON parsing and manipulation."""
    print("\n" + "=" * 60)
    print("WORKING WITH JSON")
    print("=" * 60)
    
    print("\n1. Parsing JSON Responses:")
    # Example JSON response (simulated)
    json_response = '''
    {
        "status": "success",
        "data": {
            "user": {
                "id": 12345,
                "name": "Rem Bot",
                "email": "rem@example.com",
                "skills": ["Python", "API", "Automation"],
                "stats": {
                    "projects": 15,
                    "contributions": 42,
                    "stars": 7
                }
            },
            "timestamp": "2026-02-15T13:06:00Z"
        }
    }
    '''
    
    try:
        # Parse JSON string
        data = json.loads(json_response)
        print("  Parsed JSON structure:")
        print(f"    Status: {data.get('status')}")
        print(f"    User: {data['data']['user']['name']}")
        print(f"    Email: {data['data']['user']['email']}")
        print(f"    Skills: {', '.join(data['data']['user']['skills'])}")
        print(f"    Projects: {data['data']['user']['stats']['projects']}")
        
        # Convert Python object to JSON string
        python_dict = {
            "name": "Test Object",
            "values": [1, 2, 3, 4, 5],
            "nested": {
                "enabled": True,
                "count": 42
            }
        }
        
        json_string = json.dumps(python_dict, indent=2)
        print(f"\n  Python dict converted to JSON:")
        print(f"    Type: {type(json_string)}")
        print(f"    Length: {len(json_string)} characters")
        
    except json.JSONDecodeError as e:
        print(f"  JSON parsing error: {e}")
    
    print("\n2. Saving and Loading JSON Files:")
    
    # Create sample data
    sample_data = {
        "timestamp": datetime.now().isoformat(),
        "api_requests_made": 5,
        "endpoints_called": ["/users", "/posts", "/search"],
        "config": {
            "timeout": 30,
            "retry_count": 3,
            "user_agent": "Python-Learning-Bot/1.0"
        }
    }
    
    # Save to file
    filename = "api_log.json"
    try:
        with open(filename, "w") as f:
            json.dump(sample_data, f, indent=2)
        print(f"  ✓ Saved sample data to {filename}")
        
        # Load from file
        with open(filename, "r") as f:
            loaded_data = json.load(f)
        print(f"  ✓ Loaded data from {filename}")
        print(f"    Timestamp: {loaded_data.get('timestamp')}")
        print(f"    Requests made: {loaded_data.get('api_requests_made')}")
        
    except Exception as e:
        print(f"  ✗ Error with JSON file operations: {e}")
    
    # Cleanup
    import os
    if os.path.exists(filename):
        os.remove(filename)
        print(f"  ✓ Cleaned up {filename}")


def error_handling_and_headers():
    """Demonstrate error handling and HTTP headers."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING AND HEADERS")
    print("=" * 60)
    
    print("\n1. Comprehensive Error Handling:")
    
    test_urls = [
        ("Valid API", "https://jsonplaceholder.typicode.com/todos/1"),
        ("Invalid URL", "https://invalid-url-that-does-not-exist.xyz"),
        ("Non-existent endpoint", "https://jsonplaceholder.typicode.com/nonexistent"),
        ("Timeout test", "https://httpbin.org/delay/5")  # 5-second delay
    ]
    
    for name, url in test_urls:
        print(f"\n  Testing: {name}")
        print(f"  URL: {url}")
        
        try:
            if "Timeout" in name:
                response = requests.get(url, timeout=2)  # Short timeout
            else:
                response = requests.get(url, timeout=10)
            
            print(f"    Status: {response.status_code} {response.reason}")
            
            if response.status_code == 200:
                print(f"    Success! Response preview: {response.text[:50]}...")
            elif response.status_code == 404:
                print(f"    Resource not found")
            else:
                print(f"    Unexpected status")
                
        except requests.exceptions.Timeout:
            print(f"    ⏱️  Timeout: Request took too long")
        except requests.exceptions.ConnectionError:
            print(f"    🔌 Connection Error: Could not reach server")
        except requests.exceptions.RequestException as e:
            print(f"    ❌ Request Exception: {e}")
        except Exception as e:
            print(f"    ⚠️  Unexpected error: {e}")
    
    print("\n2. Working with HTTP Headers:")
    
    headers = {
        "User-Agent": "Python-Learning-Bot/1.0",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache"
    }
    
    test_url = "https://httpbin.org/headers"
    
    try:
        response = requests.get(test_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"  Your headers were sent as:")
            for key, value in data.get('headers', {}).items():
                if key.lower().startswith('user-agent') or key.lower().startswith('accept'):
                    print(f"    {key}: {value}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n3. Rate Limiting Awareness:")
    print("  Important: Always respect API rate limits!")
    print("  Tips:")
    print("    • Check API documentation for rate limits")
    print("    • Implement delays between requests")
    print("    • Use caching when possible")
    print("    • Handle 429 (Too Many Requests) errors gracefully")
    
    # Demonstrate adding delay
    print(f"\n  Adding 1 second delay between batch requests...")
    for i in range(3):
        time.sleep(1)  # 1 second delay
        print(f"    Request {i+1} at {datetime.now().strftime('%H:%M:%S')}")


def practical_api_projects():
    """Practical API integration examples."""
    print("\n" + "=" * 60)
    print("PRACTICAL API PROJECTS")
    print("=" * 60)
    
    print("\n1. News Headlines Fetcher:")
    
    # Using NewsAPI example (would require API key in real use)
    print("  Example using NewsAPI (requires API key):")
    print("""
  import requests
  
  API_KEY = "your_api_key_here"
  url = "https://newsapi.org/v2/top-headlines"
  
  params = {
      "country": "us",
      "category": "technology",
      "apiKey": API_KEY,
      "pageSize": 5
  }
  
  response = requests.get(url, params=params)
  if response.status_code == 200:
      articles = response.json().get('articles', [])
      for article in articles:
          print(f"Title: {article.get('title')}")
          print(f"Source: {article.get('source', {}).get('name')}")
          print(f"URL: {article.get('url')}")
          print("-" * 40)
  """)
    
    print("\n2. Currency Converter:")
    
    # Using exchangerate-api.com (free tier)
    print("  Using exchangerate-api.com (free, no API key required):")
    
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        if response.status_code == 200:
            data = response.json()
            base_currency = data.get('base', 'USD')
            rates = data.get('rates', {})
            
            print(f"  Base currency: {base_currency}")
            print(f"  Last updated: {data.get('date')}")
            print(f"  Sample exchange rates:")
            for currency in ['EUR', 'GBP', 'JPY', 'CAD', 'AUD'][:3]:
                rate = rates.get(currency)
                if rate:
                    print(f"    1 {base_currency} = {rate:.3f} {currency}")
    except Exception as e:
        print(f"  Error fetching rates: {e}")
    
    print("\n3. Build Your Own API Client Class:")
    
    print("""
  class GitHubClient:
      def __init__(self, token=None):
          self.base_url = "https://api.github.com"
          self.headers = {"Accept": "application/vnd.github.v3+json"}
          if token:
              self.headers["Authorization"] = f"token {token}"
      
      def get_user(self, username):
          url = f"{self.base_url}/users/{username}"
          response = requests.get(url, headers=self.headers)
          response.raise_for_status()
          return response.json()
      
      def get_repos(self, username):
          url = f"{self.base_url}/users/{username}/repos"
          response = requests.get(url, headers=self.headers)
          response.raise_for_status()
          return response.json()
  """)
    
    print("\n4. Best Practices:")
    print("""
  1. Always use try-except blocks for network requests
  2. Set reasonable timeouts (e.g., timeout=10)
  3. Respect rate limits and implement retry logic
  4. Cache responses when appropriate
  5. Use environment variables for API keys
  6. Validate and sanitize input parameters
  7. Log API calls for debugging
  8. Handle different HTTP status codes appropriately
  """)


def main():
    """Run all API demonstrations."""
    print("\n" + "=" * 60)
    print("PYTHON API TUTORIAL")
    print("=" * 60)
    print("This tutorial demonstrates working with REST APIs in Python")
    print("using the requests library for HTTP communication.\n")
    
    basic_api_request()
    api_with_parameters()
    working_with_json()
    error_handling_and_headers()
    practical_api_projects()
    
    print("\n" + "=" * 60)
    print("GETTING STARTED WITH APIs")
    print("=" * 60)
    print("""
To run these examples:
1. Install requests: pip install requests
2. Run this file: python api_example.py
3. Modify examples to use real APIs with your own keys

Free APIs to Practice With:
• JSONPlaceholder - Fake REST API (no auth)
• OpenWeatherMap - Weather data (free tier)
• NewsAPI - News headlines (free tier)
• GitHub API - Repository/user info (authenticated)
• ExchangeRate-API - Currency rates (free tier)
• httpbin.org - HTTP request testing
    """)
    
    print("\n⚠️  Important Notes:")
    print("• Always read API documentation before use")
    print("• Never commit API keys to version control")
    print("• Respect rate limits and terms of service")
    print("• Handle errors gracefully in production code")
    
    print("\n✅ API tutorial completed successfully!")


if __name__ == "__main__":
    main()