# Benchmark Repository

This git repository is for the benchmarking of reranking and embedding APIs.

## Setup

In order to setup this repo, ensure to install dependencies:

- [Astral UV](https://docs.astral.sh/uv/)

```python
uv sync
```

Then, you will want to download the data (queries and documents).

```python
python src/download_data.py
```

Finally, you will want to specify the necessary environment variables,

```bash
cp .env.example .env
vim .env
```

## Running the benchmark

To test the latency and throughput of the endpoint, you will want to do a sweep over QPS to explore the latencies you get at different amounts of traffic. Then, you can graph the results with [graph_tests.ipynb](/src/graph_tests.ipynb)

```python
python src/benchmark.py --provider zeroentropy --task rerank --qps 1 --duration 30 --save logs/rerank-qps-1.json
python src/benchmark.py --provider zeroentropy --task rerank --qps 2 --duration 30 --save logs/rerank-qps-2.json
python src/benchmark.py --provider zeroentropy --task rerank --qps 3 --duration 30 --save logs/rerank-qps-3.json
python src/benchmark.py --provider zeroentropy --task rerank --qps 4 --duration 15 --save logs/rerank-qps-4.json
python src/benchmark.py --provider zeroentropy --task rerank --qps 6 --duration 15 --save logs/rerank-qps-6.json
python src/benchmark.py --provider zeroentropy --task rerank --qps 8 --duration 15 --save logs/rerank-qps-8.json
python src/benchmark.py --provider zeroentropy --task rerank --qps 12 --duration 15 --save logs/rerank-qps-12.json

python src/benchmark.py --provider zeroentropy --task embed-queries --qps 10 --duration 30 --save logs/embed-queries-qps-10.json
python src/benchmark.py --provider zeroentropy --task embed-queries --qps 50 --duration 30 --save logs/embed-queries-qps-50.json
python src/benchmark.py --provider zeroentropy --task embed-queries --qps 100 --duration 30 --save logs/embed-queries-qps-100.json
python src/benchmark.py --provider zeroentropy --task embed-queries --qps 200 --duration 30 --save logs/embed-queries-qps-200.json
python src/benchmark.py --provider zeroentropy --task embed-queries --qps 300 --duration 30 --save logs/embed-queries-qps-300.json
python src/benchmark.py --provider zeroentropy --task embed-queries --qps 500 --duration 15 --save logs/embed-queries-qps-500.json
```