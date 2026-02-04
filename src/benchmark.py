import asyncio
import json
import os
import statistics
import time
from random import Random
from typing import Literal
from uuid import uuid4

import cohere
import httpx
import numpy as np
import typed_argparse as tap
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

sem = asyncio.Semaphore(256)

DEFAULT_K = 50
DEBUG = False


class Args(tap.TypedArgs):
    provider: Literal["zeroentropy", "cohere", "jina"] = tap.arg(
        default="zeroentropy",
        help="The provider to test.",
    )
    task: Literal["rerank", "embed-queries", "embed-documents"] = tap.arg(
        help="The task to benchmark",
    )
    qps: float = tap.arg(
        default=0.5,
        help="QPS to test with",
    )
    duration: float = tap.arg(
        default=10.0,
        help="The number of seconds to test for",
    )
    k: int | None = tap.arg(
        default=None,
        help=f'Number of items to rerank/embed. If not provided, will be 1 if the task is "embed-queries", and {DEFAULT_K} otherwise.',
    )
    save: str | None = tap.arg(
        default=None,
        help="The location to save the times to",
    )


http_client = httpx.AsyncClient(
    http2=True,
    limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
    timeout=30,
)
ZEROENTROPY_BASE_URL = os.getenv("ZEROENTROPY_BASE_URL")
ZEROENTROPY_API_KEY = os.getenv("ZEROENTROPY_API_KEY")
co = cohere.AsyncClientV2()
jina = cohere.AsyncClient(
    base_url="https://api.jina.ai",
    api_key=os.getenv("JINA_API_KEY"),
)


async def call_reranker(
    provider: Literal["zeroentropy", "cohere", "jina"],
    query: str,
    documents: list[str],
) -> float:
    query = str(uuid4()) + query
    documents = [str(uuid4()) + document for document in documents]
    start_time = time.perf_counter()
    async with sem:
        match provider:
            case "zeroentropy":
                response = await http_client.post(
                    f"{ZEROENTROPY_BASE_URL}/models/rerank",
                    headers={
                        "Authorization": f"Bearer {ZEROENTROPY_API_KEY}",
                    },
                    json={
                        "model": "zerank-1",
                        "query": query,
                        "documents": documents,
                        "latency_mode": "fast",
                    },
                )
                response.raise_for_status()
            case "cohere":
                response = await co.rerank(
                    model="rerank-v3.5",
                    query=query,
                    documents=documents,
                )
            case "jina":
                response = await jina.rerank(
                    model="jina-reranker-m0",
                    query=query,
                    documents=documents,
                )
    end_time = time.perf_counter()
    if DEBUG:
        print(f"Time: {1000 * (end_time - start_time):.1f}ms")
    return end_time - start_time


async def call_embedding(
    provider: Literal["zeroentropy", "cohere", "jina"],
    embedding_type: Literal["query", "document"],
    texts: list[str],
) -> float:
    texts = [str(uuid4()) + text for text in texts]
    start_time = time.perf_counter()
    async with sem:
        match provider:
            case "zeroentropy":
                response = await http_client.post(
                    f"{ZEROENTROPY_BASE_URL}/models/embed",
                    headers={
                        "Authorization": f"Bearer {ZEROENTROPY_API_KEY}",
                    },
                    json={
                        "model": "qwen/qwen3-4b",
                        "embedding_type": embedding_type,
                        "input": texts,
                        "latency_mode": "fast",
                    },
                )
                response.raise_for_status()
            case "cohere":
                response = await co.embed(
                    model="rerank-v3.5",
                    input_type=embedding_type,
                    texts=texts,
                )
            case "jina":
                response = await jina.embed(
                    model="jina-embeddings-v4",
                    input_type=embedding_type,
                    texts=texts,
                )
    end_time = time.perf_counter()
    if DEBUG:
        print(f"Time: {1000 * (end_time - start_time):.1f}ms")
    return end_time - start_time


async def main(args: Args) -> None:
    if args.k is None:
        k = 1 if args.task == "embed-queries" else DEFAULT_K
    else:
        k = args.k
    # Use the same seed for fair comparison between models
    rng = Random(42)

    # Setup input
    with open("data/queries.json") as f:
        queries_json = json.loads(f.read())
    with open("data/documents.json") as f:
        documents_json = json.loads(f.read())
    queries: list[str] = queries_json
    documents: list[str] = documents_json
    documents = [d + d[:500] for d in documents]

    # Warmup the endpoint / HTTPS connection
    match args.task:
        case "rerank":
            await call_reranker(args.provider, "What is 2+2?", ["4", "1 million"])
        case "embed-documents":
            await call_embedding(args.provider, "document", ["4", "1 million"])
        case "embed-queries":
            await call_embedding(args.provider, "query", ["4", "1 million"])

    # Start running tasks as a poisson distribution
    delays = np.random.exponential(
        1 / args.qps, size=int(args.duration * args.qps * 1.5)
    )
    send_times = np.cumsum(delays)
    send_times = send_times[send_times < args.duration]

    tasks: list[asyncio.Task[float]] = []
    start_time = time.time()
    for t in tqdm(send_times, desc="API Calls", disable=DEBUG):
        delay = t - (time.time() - start_time)
        if delay > 0:
            await asyncio.sleep(delay)

        match args.task:
            case "rerank":
                chosen_query = rng.choice(queries)
                chosen_documents = rng.sample(documents, k)
                tasks.append(
                    asyncio.create_task(
                        call_reranker(args.provider, chosen_query, chosen_documents)
                    )
                )
            case "embed-documents":
                chosen_documents = rng.sample(documents, k)
                tasks.append(
                    asyncio.create_task(
                        call_embedding(args.provider, "document", chosen_documents)
                    )
                )
            case "embed-queries":
                chosen_query = rng.sample(queries, k)
                tasks.append(
                    asyncio.create_task(
                        call_embedding(args.provider, "query", chosen_query)
                    )
                )

    latencies = await asyncio.gather(*tasks)

    mean = sum(latencies) / len(latencies)
    stddev = statistics.stdev(latencies)
    if args.save is not None:
        with open(args.save, "w") as f:
            f.write(json.dumps(latencies, indent=4))
    print(f"\nMean Time: {1000 * mean:.1f}ms ± {1000 * stddev:.1f}ms")


if __name__ == "__main__":

    def main_sync(args: Args) -> None:
        asyncio.run(main(args))

    tap.Parser(Args).bind(main_sync).run()
