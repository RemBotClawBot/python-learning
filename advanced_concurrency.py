#!/usr/bin/env python3
"""
Advanced Concurrency Examples
=============================

This module demonstrates advanced Python concurrency patterns:
- Asyncio with advanced patterns
- Thread pools and process pools
- Concurrent futures
- Parallel processing with multiprocessing
- Real-world use cases (web scraping, API calls, data processing)

Covers: async/await, semaphores, queues, timeouts, error handling,
thread synchronization, process communication, and performance comparisons.
"""

import asyncio
import concurrent.futures
import multiprocessing
import threading
import queue
import time
import random
from typing import List, Dict, Any, Callable
import math
import sys
import os
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class TaskResult:
    """Result container for task execution."""
    task_id: int
    success: bool
    result: Any
    duration: float
    error: str = ""


def demo_thread_pool():
    """
    Demonstrate ThreadPoolExecutor for I/O-bound tasks.
    Simulates fetching data from multiple URLs concurrently.
    """
    print("\n" + "="*60)
    print("THREAD POOL EXECUTOR (I/O-bound tasks)")
    print("="*60)
    
    def fetch_url(url_id: int) -> Dict[str, Any]:
        """Simulate fetching data from a URL."""
        sleep_time = random.uniform(0.1, 0.5)
        time.sleep(sleep_time)  # Simulate network delay
        
        # Simulate occasional failure
        if random.random() < 0.1:  # 10% chance of failure
            raise ConnectionError(f"Failed to fetch URL {url_id}")
        
        return {
            "url_id": url_id,
            "status": "success",
            "data": f"Content from URL {url_id}",
            "size": random.randint(100, 10000),
            "duration": sleep_time
        }
    
    num_urls = 20
    urls = list(range(num_urls))
    
    print(f"Fetching {num_urls} URLs sequentially...")
    start_time = time.time()
    sequential_results = []
    for url_id in urls:
        try:
            result = fetch_url(url_id)
            sequential_results.append(result)
        except ConnectionError as e:
            sequential_results.append({"url_id": url_id, "error": str(e)})
    sequential_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Successful fetches: {len([r for r in sequential_results if 'error' not in r])}/{num_urls}")
    
    print(f"\nFetching {num_urls} URLs concurrently with ThreadPoolExecutor...")
    start_time = time.time()
    concurrent_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(fetch_url, url_id): url_id for url_id in urls}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            url_id = future_to_url[future]
            try:
                result = future.result()
                concurrent_results.append(result)
            except Exception as e:
                concurrent_results.append({"url_id": url_id, "error": str(e)})
    
    concurrent_time = time.time() - start_time
    
    print(f"Concurrent time: {concurrent_time:.2f}s")
    print(f"Successful fetches: {len([r for r in concurrent_results if 'error' not in r])}/{num_urls}")
    print(f"Speedup: {sequential_time/concurrent_time:.2f}x")
    
    return sequential_time, concurrent_time


def demo_process_pool():
    """
    Demonstrate ProcessPoolExecutor for CPU-bound tasks.
    Simulates computing prime numbers in parallel.
    """
    print("\n" + "="*60)
    print("PROCESS POOL EXECUTOR (CPU-bound tasks)")
    print("="*60)
    
    def is_prime(n: int) -> bool:
        """Check if a number is prime (CPU-intensive)."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Check odd divisors up to sqrt(n)
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def count_primes_in_range(start: int, end: int) -> int:
        """Count primes in a range (CPU-bound task)."""
        count = 0
        for n in range(start, end + 1):
            if is_prime(n):
                count += 1
        return count
    
    # Generate ranges to process
    ranges = []
    batch_size = 5000
    total_numbers = 100000
    
    for i in range(0, total_numbers, batch_size):
        ranges.append((i, min(i + batch_size - 1, total_numbers)))
    
    print(f"Counting primes in ranges 0-{total_numbers} (batch size: {batch_size})")
    print(f"Number of batches: {len(ranges)}")
    
    # Sequential execution
    print("\nCounting primes sequentially...")
    start_time = time.time()
    sequential_total = 0
    for start, end in ranges:
        sequential_total += count_primes_in_range(start, end)
    sequential_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Total primes found: {sequential_total}")
    
    # Parallel execution
    print("\nCounting primes in parallel with ProcessPoolExecutor...")
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        futures = [executor.submit(count_primes_in_range, start, end) 
                   for start, end in ranges]
        
        # Collect results
        parallel_total = 0
        for future in concurrent.futures.as_completed(futures):
            parallel_total += future.result()
    
    parallel_time = time.time() - start_time
    
    print(f"Parallel time: {parallel_time:.2f}s")
    print(f"Total primes found: {parallel_total}")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    
    return sequential_time, parallel_time


async def demo_advanced_asyncio():
    """
    Demonstrate advanced asyncio patterns:
    - Semaphores for rate limiting
    - Queues for task distribution
    - Timeouts and cancellation
    - Error handling with retries
    """
    print("\n" + "="*60)
    print("ADVANCED ASYNCIO PATTERNS")
    print("="*60)
    
    class AsyncRateLimitedFetcher:
        """Rate-limited async fetcher using semaphores."""
        
        def __init__(self, max_concurrent: int = 3, retries: int = 2):
            self.semaphore = asyncio.Semaphore(max_concurrent)
            self.retries = retries
        
        async def fetch_with_backoff(self, task_id: int) -> TaskResult:
            """Fetch with exponential backoff and retries."""
            start_time = time.time()
            
            for attempt in range(self.retries + 1):
                try:
                    async with self.semaphore:
                        # Simulate async fetch with random delay
                        await asyncio.sleep(random.uniform(0.05, 0.2))
                        
                        # Simulate occasional failure
                        if random.random() < 0.2:  # 20% failure rate
                            raise ConnectionError(f"Network error on task {task_id}")
                        
                        result = f"Data from task {task_id} (attempt {attempt + 1})"
                        duration = time.time() - start_time
                        
                        return TaskResult(
                            task_id=task_id,
                            success=True,
                            result=result,
                            duration=duration
                        )
                
                except Exception as e:
                    if attempt == self.retries:
                        duration = time.time() - start_time
                        return TaskResult(
                            task_id=task_id,
                            success=False,
                            result=None,
                            duration=duration,
                            error=str(e)
                        )
                    
                    # Exponential backoff
                    backoff = (2 ** attempt) * 0.1
                    await asyncio.sleep(backoff)
    
    async def producer_consumer_queue():
        """Demonstrate producer-consumer pattern with asyncio.Queue."""
        print("\nProducer-Consumer Pattern with asyncio.Queue:")
        
        queue = asyncio.Queue(maxsize=10)
        results = []
        
        async def producer(num_items: int):
            """Produce items for the queue."""
            for i in range(num_items):
                await asyncio.sleep(random.uniform(0.01, 0.05))
                await queue.put(f"Item_{i}")
                # print(f"Produced: Item_{i}")
            await queue.put(None)  # Sentinel value
        
        async def consumer(consumer_id: int):
            """Consume items from the queue."""
            while True:
                item = await queue.get()
                if item is None:
                    # Put sentinel back for other consumers
                    await queue.put(None)
                    break
                
                # Simulate processing
                await asyncio.sleep(random.uniform(0.02, 0.1))
                result = f"Consumer_{consumer_id} processed {item}"
                results.append(result)
                # print(result)
                queue.task_done()
        
        # Run producer and multiple consumers
        num_items = 20
        num_consumers = 3
        
        producer_task = asyncio.create_task(producer(num_items))
        consumer_tasks = [asyncio.create_task(consumer(i)) for i in range(num_consumers)]
        
        await asyncio.gather(producer_task)
        await queue.join()
        
        # Cancel consumers
        for task in consumer_tasks:
            task.cancel()
        
        print(f"Produced {num_items} items")
        print(f"Processed by {num_consumers} consumers")
        print(f"Total results: {len(results)}")
        return results
    
    async def timeout_and_cancellation():
        """Demonstrate timeout and task cancellation."""
        print("\nTimeout and Cancellation:")
        
        async def slow_task(task_id: int, delay: float) -> str:
            """A task that takes some time."""
            try:
                await asyncio.sleep(delay)
                return f"Task {task_id} completed after {delay}s"
            except asyncio.CancelledError:
                print(f"Task {task_id} was cancelled!")
                raise
        
        # Task with timeout
        print("Demonstrating timeout...")
        try:
            result = await asyncio.wait_for(slow_task(1, 2.0), timeout=1.0)
            print(f"Result: {result}")
        except asyncio.TimeoutError:
            print("Task 1 timed out after 1 second")
        
        # Task cancellation
        print("\nDemonstrating cancellation...")
        task = asyncio.create_task(slow_task(2, 5.0))
        await asyncio.sleep(0.5)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            print("Task 2 was successfully cancelled")
    
    # Run rate-limited fetcher
    print("Rate-limited fetching with semaphores:")
    fetcher = AsyncRateLimitedFetcher(max_concurrent=3, retries=2)
    
    tasks = [fetcher.fetch_with_backoff(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    successful = sum(1 for r in results if r.success)
    total_time = sum(r.duration for r in results)
    
    print(f"Successful tasks: {successful}/{len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per task: {total_time/len(results):.2f}s")
    
    # Run producer-consumer demo
    queue_results = await producer_consumer_queue()
    
    # Run timeout/cancellation demo
    await timeout_and_cancellation()
    
    return results, queue_results


def demo_multiprocessing_communication():
    """
    Demonstrate inter-process communication with multiprocessing.
    Shows pipes, queues, and shared memory.
    """
    print("\n" + "="*60)
    print("MULTIPROCESSING COMMUNICATION PATTERNS")
    print("="*60)
    
    def worker_with_queue(input_queue: multiprocessing.Queue, 
                         output_queue: multiprocessing.Queue,
                         worker_id: int):
        """Worker process that processes items from a queue."""
        while True:
            try:
                item = input_queue.get(timeout=1)
                if item is None:  # Sentinel value
                    break
                
                # Process item (simulate work)
                time.sleep(random.uniform(0.01, 0.1))
                result = f"Worker {worker_id} processed: {item * 2}"
                output_queue.put(result)
                
            except queue.Empty:
                break
    
    def worker_with_pipe(conn, worker_id: int):
        """Worker process that uses pipes for communication."""
        while True:
            try:
                message = conn.recv()
                if message == "STOP":
                    break
                
                # Process message
                time.sleep(random.uniform(0.01, 0.05))
                response = f"Worker {worker_id} received: {message}"
                conn.send(response)
                
            except EOFError:
                break
    
    # Queue-based communication
    print("Queue-based communication:")
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    
    # Start worker processes
    num_workers = 3
    workers = []
    for i in range(num_workers):
        worker = multiprocessing.Process(
            target=worker_with_queue,
            args=(input_queue, output_queue, i)
        )
        worker.start()
        workers.append(worker)
    
    # Send work to workers
    num_items = 10
    for i in range(num_items):
        input_queue.put(i)
    
    # Send termination signals
    for _ in range(num_workers):
        input_queue.put(None)
    
    # Collect results
    results = []
    for _ in range(num_items):
        try:
            result = output_queue.get(timeout=2)
            results.append(result)
        except queue.Empty:
            break
    
    # Wait for workers to finish
    for worker in workers:
        worker.join(timeout=1)
    
    print(f"Sent {num_items} items")
    print(f"Received {len(results)} results")
    
    # Pipe-based communication
    print("\nPipe-based communication:")
    parent_conns = []
    child_conns = []
    pipe_workers = []
    
    for i in range(2):
        parent_conn, child_conn = multiprocessing.Pipe()
        parent_conns.append(parent_conn)
        
        worker = multiprocessing.Process(
            target=worker_with_pipe,
            args=(child_conn, i)
        )
        worker.start()
        pipe_workers.append(worker)
    
    # Send messages via pipes
    messages = ["Hello", "World", "Python", "Multiprocessing"]
    responses = []
    
    for i, message in enumerate(messages):
        conn = parent_conns[i % len(parent_conns)]
        conn.send(message)
        
        # Receive response
        if conn.poll(timeout=1):
            response = conn.recv()
            responses.append(response)
    
    # Stop workers
    for conn in parent_conns:
        conn.send("STOP")
    
    for worker in pipe_workers:
        worker.join(timeout=1)
    
    print(f"Sent {len(messages)} messages")
    print(f"Received {len(responses)} responses")
    
    return results, responses


def demo_thread_synchronization():
    """
    Demonstrate thread synchronization primitives:
    - Locks for shared resource protection
    - Events for thread coordination
    - Conditions for complex synchronization
    """
    print("\n" + "="*60)
    print("THREAD SYNCHRONIZATION PRIMITIVES")
    print("="*60)
    
    class BankAccount:
        """Bank account with thread-safe operations."""
        
        def __init__(self, initial_balance: float = 0):
            self.balance = initial_balance
            self.lock = threading.Lock()
            self.transaction_count = 0
        
        def deposit(self, amount: float) -> bool:
            """Thread-safe deposit."""
            with self.lock:
                self.balance += amount
                self.transaction_count += 1
                print(f"Deposited ${amount:.2f}, new balance: ${self.balance:.2f}")
                return True
        
        def withdraw(self, amount: float) -> bool:
            """Thread-safe withdrawal."""
            with self.lock:
                if self.balance >= amount:
                    self.balance -= amount
                    self.transaction_count += 1
                    print(f"Withdrew ${amount:.2f}, new balance: ${self.balance:.2f}")
                    return True
                else:
                    print(f"Failed to withdraw ${amount:.2f}, insufficient funds")
                    return False
        
        def get_balance(self) -> float:
            """Thread-safe balance check."""
            with self.lock:
                return self.balance
        
        def get_transaction_count(self) -> int:
            """Thread-safe transaction count."""
            with self.lock:
                return self.transaction_count
    
    def account_worker(account: BankAccount, 
                      worker_id: int, 
                      num_transactions: int,
                      start_event: threading.Event):
        """Worker thread that performs transactions."""
        # Wait for start signal
        start_event.wait()
        
        for _ in range(num_transactions):
            # Random deposit or withdrawal
            if random.random() < 0.6:  # 60% deposit
                amount = random.uniform(10, 100)
                account.deposit(amount)
            else:  # 40% withdrawal
                amount = random.uniform(10, 100)
                account.withdraw(amount)
            
            # Small delay
            time.sleep(random.uniform(0.001, 0.01))
    
    # Create bank account
    account = BankAccount(initial_balance=1000)
    start_event = threading.Event()
    
    # Create and start worker threads
    num_workers = 5
    num_transactions_per_worker = 20
    
    workers = []
    for i in range(num_workers):
        worker = threading.Thread(
            target=account_worker,
            args=(account, i, num_transactions_per_worker, start_event)
        )
        workers.append(worker)
        worker.start()
    
    print(f"Starting {num_workers} worker threads...")
    print(f"Initial balance: ${account.get_balance():.2f}")
    
    # Signal all threads to start
    start_event.set()
    
    # Wait for all threads to complete
    for worker in workers:
        worker.join()
    
    print(f"Final balance: ${account.get_balance():.2f}")
    print(f"Total transactions: {account.get_transaction_count()}")
    
    # Demonstrate Event and Condition
    print("\nEvent and Condition synchronization:")
    
    class MessageQueue:
        """Thread-safe message queue using Condition."""
        
        def __init__(self, max_size: int = 10):
            self.queue = queue.Queue(maxsize=max_size)
            self.condition = threading.Condition()
            self.is_running = True
        
        def put(self, message: str):
            """Put message in queue, notify waiting threads."""
            with self.condition:
                self.queue.put(message)
                self.condition.notify_all()
        
        def get(self) -> str:
            """Get message from queue, wait if empty."""
            with self.condition:
                while self.queue.empty() and self.is_running:
                    self.condition.wait()
                
                if not self.is_running and self.queue.empty():
                    raise queue.Empty()
                
                return self.queue.get()
        
        def stop(self):
            """Stop the queue."""
            with self.condition:
                self.is_running = False
                self.condition.notify_all()
    
    # Test message queue
    msg_queue = MessageQueue(max_size=5)
    
    def producer_thread():
        """Produce messages."""
        for i in range(10):
            msg_queue.put(f"Message {i}")
            time.sleep(random.uniform(0.01, 0.05))
        msg_queue.stop()
    
    def consumer_thread(consumer_id: int):
        """Consume messages."""
        try:
            while True:
                message = msg_queue.get()
                print(f"Consumer {consumer_id} got: {message}")
        except queue.Empty:
            pass
    
    # Start threads
    producer = threading.Thread(target=producer_thread)
    consumers = [threading.Thread(target=consumer_thread, args=(i,)) 
                 for i in range(2)]
    
    producer.start()
    for consumer in consumers:
        consumer.start()
    
    producer.join()
    for consumer in consumers:
        consumer.join()
    
    return account.get_balance(), account.get_transaction_count()


def demo_performance_comparison():
    """
    Compare performance of different concurrency approaches
    for different types of tasks.
    """
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    def cpu_bound_task(n: int) -> float:
        """CPU-bound task: compute sum of squares."""
        return sum(i * i for i in range(n))
    
    def io_bound_task(task_id: int) -> str:
        """I/O-bound task: simulate network delay."""
        time.sleep(0.01)  # Simulate network I/O
        return f"Result from task {task_id}"
    
    num_tasks = 100
    n = 10000  # For CPU-bound task
    
    print("Comparing approaches for I/O-bound tasks:")
    
    # Sequential
    print("\n1. Sequential execution:")
    start = time.time()
    results = []
    for i in range(num_tasks):
        results.append(io_bound_task(i))
    sequential_time = time.time() - start
    print(f"   Time: {sequential_time:.3f}s")
    
    # ThreadPoolExecutor
    print("\n2. ThreadPoolExecutor (max_workers=10):")
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(io_bound_task, i) for i in range(num_tasks)]
        results = [f.result() for f in futures]
    threadpool_time = time.time() - start
    print(f"   Time: {threadpool_time:.3f}s")
    print(f"   Speedup: {sequential_time/threadpool_time:.2f}x")
    
    # Multiprocessing
    print("\n3. ProcessPoolExecutor (max_workers=4):")
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(io_bound_task, i) for i in range(num_tasks)]
        results = [f.result() for f in futures]
    processpool_time = time.time() - start
    print(f"   Time: {processpool_time:.3f}s")
    print(f"   Speedup: {sequential_time/processpool_time:.2f}x")
    
    print("\nComparing approaches for CPU-bound tasks:")
    
    # Sequential CPU-bound
    print("\n1. Sequential execution:")
    start = time.time()
    results = []
    for i in range(num_tasks):
        results.append(cpu_bound_task(n))
    sequential_cpu_time = time.time() - start
    print(f"   Time: {sequential_cpu_time:.3f}s")
    
    # ThreadPoolExecutor CPU-bound (not ideal)
    print("\n2. ThreadPoolExecutor (max_workers=10) - NOT ideal for CPU-bound:")
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(cpu_bound_task, n) for i in range(num_tasks)]
        results = [f.result() for f in futures]
    threadpool_cpu_time = time.time() - start
    print(f"   Time: {threadpool_cpu_time:.3f}s")
    print(f"   Speedup: {sequential_cpu_time/threadpool_cpu_time:.2f}x")
    
    # ProcessPoolExecutor CPU-bound
    print("\n3. ProcessPoolExecutor (max_workers=4):")
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cpu_bound_task, n) for i in range(num_tasks)]
        results = [f.result() for f in futures]
    processpool_cpu_time = time.time() - start
    print(f"   Time: {processpool_cpu_time:.3f}s")
    print(f"   Speedup: {sequential_cpu_time/processpool_cpu_time:.2f}x")
    
    return {
        'io_bound': {
            'sequential': sequential_time,
            'threadpool': threadpool_time,
            'processpool': processpool_time
        },
        'cpu_bound': {
            'sequential': sequential_cpu_time,
            'threadpool': threadpool_cpu_time,
            'processpool': processpool_cpu_time
        }
    }


def main():
    """Run all concurrency demonstrations."""
    print("="*60)
    print("ADVANCED CONCURRENCY EXAMPLES")
    print("="*60)
    print("\nThis module demonstrates Python concurrency patterns:")
    print("1. ThreadPoolExecutor for I/O-bound tasks")
    print("2. ProcessPoolExecutor for CPU-bound tasks")
    print("3. Advanced asyncio patterns")
    print("4. Multiprocessing communication")
    print("5. Thread synchronization primitives")
    print("6. Performance comparison")
    print("="*60)
    
    results = {}
    
    try:
        print("\n" + "="*60)
        print("1. THREAD POOL EXECUTOR")
        print("="*60)
        seq_time, conc_time = demo_thread_pool()
        results['thread_pool'] = {
            'sequential_time': seq_time,
            'concurrent_time': conc_time,
            'speedup': seq_time/conc_time if conc_time > 0 else 0
        }
    except Exception as e:
        print(f"Thread pool demo failed: {e}")
    
    try:
        print("\n" + "="*60)
        print("2. PROCESS POOL EXECUTOR")
        print("="*60)
        seq_time, par_time = demo_process_pool()
        results['process_pool'] = {
            'sequential_time': seq_time,
            'parallel_time': par_time,
            'speedup': seq_time/par_time if par_time > 0 else 0
        }
    except Exception as e:
        print(f"Process pool demo failed: {e}")
    
    try:
        print("\n" + "="*60)
        print("3. ADVANCED ASYNCIO PATTERNS")
        print("="*60)
        asyncio_results = asyncio.run(demo_advanced_asyncio())
        results['asyncio'] = asyncio_results
    except Exception as e:
        print(f"Asyncio demo failed: {e}")
    
    try:
        print("\n" + "="*60)
        print("4. MULTIPROCESSING COMMUNICATION")
        print("="*60)
        queue_results, pipe_results = demo_multiprocessing_communication()
        results['multiprocessing'] = {
            'queue_results_count': len(queue_results),
            'pipe_results_count': len(pipe_results)
        }
    except Exception as e:
        print(f"Multiprocessing demo failed: {e}")
    
    try:
        print("\n" + "="*60)
        print("5. THREAD SYNCHRONIZATION")
        print("="*60)
        final_balance, transaction_count = demo_thread_synchronization()
        results['thread_sync'] = {
            'final_balance': final_balance,
            'transaction_count': transaction_count
        }
    except Exception as e:
        print(f"Thread synchronization demo failed: {e}")
    
    try:
        print("\n" + "="*60)
        print("6. PERFORMANCE COMPARISON")
        print("="*60)
        perf_results = demo_performance_comparison()
        results['performance'] = perf_results
    except Exception as e:
        print(f"Performance comparison failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nKey takeaways:")
    print("1. Use ThreadPoolExecutor for I/O-bound tasks (network, file operations)")
    print("2. Use ProcessPoolExecutor for CPU-bound tasks (computations)")
    print("3. Use asyncio for high-concurrency async I/O")
    print("4. Use multiprocessing for true parallelism across CPU cores")
    print("5. Use synchronization primitives (Lock, Event, Condition) for thread safety")
    print("6. Choose the right tool for your workload type")
    
    print("\nPerformance comparison summary:")
    if 'performance' in results:
        perf = results['performance']
        
        print("\nI/O-bound tasks (network sim):")
        io = perf['io_bound']
        print(f"  Sequential: {io['sequential']:.3f}s")
        print(f"  ThreadPool: {io['threadpool']:.3f}s ({io['sequential']/io['threadpool']:.2f}x speedup)")
        print(f"  ProcessPool: {io['processpool']:.3f}s ({io['sequential']/io['processpool']:.2f}x speedup)")
        
        print("\nCPU-bound tasks (computations):")
        cpu = perf['cpu_bound']
        print(f"  Sequential: {cpu['sequential']:.3f}s")
        print(f"  ThreadPool: {cpu['threadpool']:.3f}s ({cpu['sequential']/cpu['threadpool']:.2f}x speedup)")
        print(f"  ProcessPool: {cpu['processpool']:.3f}s ({cpu['sequential']/cpu['processpool']:.2f}x speedup)")
    
    print("\n" + "="*60)
    print("All demos completed successfully!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()