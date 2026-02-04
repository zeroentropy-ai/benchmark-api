import json
import os
from typing import Any, cast

import requests
from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    load_dataset,  # pyright: ignore[reportUnknownVariableType]
)
from dotenv import load_dotenv

os.makedirs("data", exist_ok=True)

load_dotenv(override=True)

ds = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
queries = [cast(Any, row["query"]) for row, _ in zip(ds, range(1000), strict=False)]  # pyright: ignore[reportUnknownVariableType]
with open("data/queries.json", "w") as f:
    f.write(json.dumps(queries, indent=4))

response = requests.get(
    "https://gist.githubusercontent.com/npip99/159681cb97319d62a54e1eb8c58181de/raw/a6d35019ad3d728ed2ddcb182f82efa164501133/abstracts.txt"
)
response.raise_for_status()
documents = response.json()
with open("data/documents.json", "w") as f:
    f.write(json.dumps(documents, indent=4))
