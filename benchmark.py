"""
Benchmarking script to run against api server
Collects and prints metrics related to seqs per seq and latency
"""
import time
import requests
import statistics
import random
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8080/predict/batch"
TOTAL_REQUESTS = 50
CONCURRENT_USERS = 5
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 1022
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWXY"

def generate_random_sequence(length):
    return "".join(random.choice(AMINO_ACIDS) for _ in range(length))

def send_request(_):
    length = random.randint(50, MAX_SEQ_LENGTH)
    batch = [generate_random_sequence(length) for _ in range(BATCH_SIZE)]

    start = time.perf_counter()
    try:
        response = requests.post(API_URL, json={"sequences": batch}, timeout=120)
        return (time.perf_counter() - start, response.status_code == 200)
    except Exception:
        return (time.perf_counter() - start, False)

if __name__ == "__main__":
    print("Starting benchmark...")

    requests.post(API_URL, json={"sequences": ["MAPLR"]}, timeout=10)

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        results = list(executor.map(send_request, range(TOTAL_REQUESTS)))

    duration = time.perf_counter() - start_time

    latencies = [r[0] for r in results]
    successes = [r[1] for r in results]
    success_rate = (sum(successes) / TOTAL_REQUESTS) * 100
    total_seqs = TOTAL_REQUESTS * BATCH_SIZE

    print(f"\nPerformance Report  --------")
    print(f"Throughput:      {total_seqs / duration:.2f} seq/sec")
    print(f"Success Rate:    {success_rate:.1f}%")
    print(f"Avg Latency:     {statistics.mean(latencies):.3f}s")
    print(f"P95 Latency:     {statistics.quantiles(latencies, n=20)[18]:.3f}s")
    print(f"P99 Latency:     {statistics.quantiles(latencies, n=100)[98]:.3f}s")
    print(f"\n")
