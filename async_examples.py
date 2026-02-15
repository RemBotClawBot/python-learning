"""
Async/Await Examples
--------------------
Modern Python async programming patterns and examples.
Demonstrates asyncio, async/await syntax, and practical use cases.
"""

import asyncio
import time
from typing import List, Dict, TYPE_CHECKING
import json

try:
    import aiohttp  # type: ignore
except ImportError:
    aiohttp = None  # type: ignore

if TYPE_CHECKING:
    import aiohttp

# ============================================================================
# PART 1: BASIC ASYNC PATTERNS
# ============================================================================

async def simple_coroutine(name: str, delay: float):
    """Simple async coroutine with delay."""
    print(f"🔄 Starting '{name}'... waiting {delay} seconds")
    await asyncio.sleep(delay)
    print(f"✅ '{name}' completed after {delay} seconds")
    return f"Result from {name}"


async def run_parallel_coroutines():
    """Run multiple coroutines concurrently."""
    print("\n" + "="*60)
    print("Running parallel coroutines...")
    
    start_time = time.time()
    
    # Create tasks for concurrent execution
    tasks = [
        simple_coroutine("Task A", 1.5),
        simple_coroutine("Task B", 1.0),
        simple_coroutine("Task C", 2.0),
        simple_coroutine("Task D", 0.5)
    ]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time for parallel execution: {elapsed:.2f} seconds")
    print(f"Results: {results}")
    
    return results


# ============================================================================
# PART 2: ASYNC HTTP REQUESTS
# ============================================================================

async def fetch_url(session: "aiohttp.ClientSession", url: str):
    """Fetch a single URL asynchronously."""
    try:
        print(f"🌐 Fetching: {url}")
        async with session.get(url) as response:
            content_type = response.headers.get('Content-Type', 'text/html')
            content_length = len(await response.read())
            return {
                "url": url,
                "status": response.status,
                "content_type": content_type,
                "content_length": content_length,
                "success": True
            }
    except Exception as e:
        return {
            "url": url,
            "status": 0,
            "error": str(e),
            "success": False
        }


async def fetch_multiple_urls(urls: List[str]):
    """Fetch multiple URLs concurrently."""
    print("\n" + "="*60)
    print("Fetching multiple URLs asynchronously...")
    
    if aiohttp is None:
        print("⚠️  aiohttp is not installed. Skipping HTTP demo.")
        print("    Install with: pip install aiohttp")
        return [
            {
                "url": url,
                "status": None,
                "success": False,
                "error": "aiohttp not installed"
            }
            for url in urls
        ]
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Fetched {len(urls)} URLs in {elapsed:.2f} seconds")
    
    successful = sum(1 for r in results if r["success"])
    print(f"✅ Successful: {successful}/{len(urls)}")
    
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['url']} - Status: {result.get('status', 'N/A')}")
    
    return results


# ============================================================================
# PART 3: ASYNC QUEUE PATTERN
# ============================================================================

async def worker(name: str, queue: asyncio.Queue):
    """Worker coroutine that processes items from a queue."""
    while True:
        try:
            # Get item from queue with timeout
            item = await asyncio.wait_for(queue.get(), timeout=5.0)
            
            if item is None:
                print(f"🛑 {name}: No more items. Exiting.")
                queue.task_done()
                break
            
            print(f"👷 {name} processing: {item}")
            
            # Simulate work
            await asyncio.sleep(0.5)
            print(f"✅ {name} completed: {item}")
            
            queue.task_done()
            
        except asyncio.TimeoutError:
            print(f"🕒 {name}: Queue empty, stopping...")
            break


async def producer(queue: asyncio.Queue, items: List[str]):
    """Producer coroutine that adds items to a queue."""
    for i, item in enumerate(items):
        await asyncio.sleep(0.2)  # Simulate production delay
        await queue.put(item)
        print(f"📦 Produced: {item}")
    
    # Signal workers that no more items will be produced
    for _ in range(3):  # Number of workers
        await queue.put(None)


async def run_worker_pattern():
    """Demonstrate producer-consumer pattern with async queue."""
    print("\n" + "="*60)
    print("Running producer-consumer pattern...")
    
    queue = asyncio.Queue(maxsize=10)
    items = [f"Item-{i}" for i in range(1, 16)]
    
    # Start workers
    worker_tasks = [
        asyncio.create_task(worker(f"Worker-{i}", queue))
        for i in range(1, 4)
    ]
    
    # Start producer
    producer_task = asyncio.create_task(producer(queue, items))
    
    # Wait for producer to finish
    await producer_task
    
    # Wait for all items to be processed
    await queue.join()
    
    # Cancel workers (they'll timeout and exit)
    for task in worker_tasks:
        task.cancel()
    
    print("✅ All items processed")


# ============================================================================
# PART 4: ASYNC TIMEOUTS AND ERROR HANDLING
# ============================================================================

async def slow_operation(duration: float):
    """Simulate a slow operation."""
    print(f"🐌 Starting slow operation ({duration}s)...")
    await asyncio.sleep(duration)
    return f"Operation completed in {duration}s"


async def run_with_timeout():
    """Demonstrate timeout handling."""
    print("\n" + "="*60)
    print("Testing timeouts...")
    
    try:
        # This should succeed (operation is faster than timeout)
        result = await asyncio.wait_for(
            slow_operation(1.0),
            timeout=2.0
        )
        print(f"✅ Fast operation: {result}")
    except asyncio.TimeoutError:
        print("❌ Fast operation timed out!")
    
    try:
        # This should timeout (operation is slower than timeout)
        result = await asyncio.wait_for(
            slow_operation(3.0),
            timeout=1.5
        )
        print(f"✅ Slow operation: {result}")
    except asyncio.TimeoutError:
        print("❌ Slow operation timed out (as expected!)")


# ============================================================================
# PART 5: ASYNC CONTEXT MANAGERS
# ============================================================================

class AsyncDatabaseConnection:
    """Example async context manager for database connections."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False
    
    async def __aenter__(self):
        """Enter async context."""
        print(f"🔌 Connecting to: {self.connection_string}")
        await asyncio.sleep(0.5)  # Simulate connection delay
        self.connected = True
        print("✅ Connected!")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        print("🔌 Closing connection...")
        await asyncio.sleep(0.2)  # Simulate cleanup
        self.connected = False
        print("✅ Connection closed")
        
        if exc_type:
            print(f"⚠️  Exception occurred: {exc_val}")
        
        # Return False to propagate exception, True to suppress it
        return False
    
    async def query(self, sql: str):
        """Execute a query."""
        if not self.connected:
            raise RuntimeError("Not connected!")
        
        print(f"📝 Executing query: {sql}")
        await asyncio.sleep(0.3)  # Simulate query execution
        return {"result": "Sample data", "rows": 42}


async def demo_async_context():
    """Demonstrate async context manager."""
    print("\n" + "="*60)
    print("Using async context manager...")
    
    async with AsyncDatabaseConnection("postgresql://localhost/mydb") as db:
        result = await db.query("SELECT * FROM users LIMIT 10")
        print(f"📊 Query result: {result}")
    
    print("✅ Context manager demo complete")


# ============================================================================
# PART 6: ASYNC MAIN FUNCTION WITH PRACTICAL EXAMPLE
# ============================================================================

async def main_demo():
    """Main async demonstration function."""
    print("🚀 Starting Async/Await Examples")
    print("="*60)
    
    try:
        # 1. Basic parallel execution
        await run_parallel_coroutines()
        
        # 2. Async HTTP requests
        urls = [
            "https://httpbin.org/get",
            "https://httpbin.org/status/200",
            "https://httpbin.org/status/404",
            "https://httpbin.org/delay/1"
        ]
        await fetch_multiple_urls(urls)
        
        # 3. Worker pattern
        await run_worker_pattern()
        
        # 4. Timeout handling
        await run_with_timeout()
        
        # 5. Async context manager
        await demo_async_context()
        
        print("\n" + "="*60)
        print("🎉 All async examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in async demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n✨ Async programming is powerful for I/O-bound tasks!")
        print("   Use it for: web scraping, API calls, database queries,")
        print("   websockets, and any task waiting for external resources.")


# ============================================================================
# RUN THE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    if aiohttp is None:
        print("⚠️  'aiohttp' not installed. Some examples will be limited.")
        print("   Install with: pip install aiohttp")

        async def limited_demo():
            await run_parallel_coroutines()
            await run_worker_pattern()
            await run_with_timeout()
            await demo_async_context()

        asyncio.run(limited_demo())
    else:
        asyncio.run(main_demo())

    print("\n📚 Key Concepts Learned:")
    print("   1. async/await syntax -> Define asynchronous operations")
    print("   2. asyncio.run() -> Run async code from synchronous context")
    print("   3. asyncio.gather() -> Run multiple coroutines concurrently")
    print("   4. asyncio.create_task() -> Schedule coroutine execution")
    print("   5. asyncio.wait_for() -> Add timeouts to async operations")
    print("   6. Async context managers -> Resource management")
    print("   7. Queues with asyncio.Queue -> Producer-consumer patterns")
