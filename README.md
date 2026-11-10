# Modeling Connected Data with Graphs and Hypervectors

Companion notebooks for the **Connected Data London (CDL) 2026** two-hour masterclass,
led by **Prashanth Rao and David Hughes from HDC Labs**.

How do you find a useful precedent when you only know part of the story? And once a case
is resolved, how can that one example help with the next one?

In this workshop, we'll investigate a delayed delivery for a London homeware retailer.
A customer in Ireland is waiting for a cordless lamp, order **SO-1042**, shipped from the
East London depot. Support knows the product, depot and country, but not what went wrong.
We'll follow the connected records, retrieve similar past orders, and update a memory
that helps decide who to call first: the depot, customs broker or carrier.

The workshop is for graph practitioners who are comfortable with basic Python and Cypher.
**No prior knowledge of hyperdimensional computing (HDC) is needed.** We introduce
hypervectors and their operators through small, runnable examples, with figures from the
talk beside the relevant code.

> A graph models connections explicitly. Hypervectors model them implicitly.
> We'll use the graph to inspect exact evidence, and hypervectors to compare patterns
> and learn from resolved cases.

## What you'll work through

Run the notebooks in order. Each rebuilds the data it needs, so you can restart a notebook
without depending on another notebook's kernel or database.

| Notebook | What you'll do |
|---|---|
| [1 · A sales-order graph](src/01_sales_order_graph.ipynb) | Store connected records in LanceDB, query them with Cypher, and find an exact precedent. |
| [2 · The three HDC operators](src/02_hdc_operators.ipynb) | Build hypervectors and experiment with binding ⊗, bundling ⊕ and permutation ρ. |
| [3 · Motif search](src/03_motif_search.ipynb) | Compose patterns from stored edge hypervectors at query time, rank partial matches, compare with Cypher scoring, and inspect the graph evidence. |
| [4 · Learning from one example](src/04_one_shot_learning.ipynb) | Summarise cases into issue prototypes and add one newly resolved order to an existing prototype, then inspect the predictions before and after. |

This is a small, constructed teaching dataset. “One example” means **one additional
confirmed case**, not learning an entire class without prior examples. The notebooks
include the assumptions and limits alongside the results.

## Set up before the workshop

You'll need Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), and internet
access for the initial installation. uv manages the project's Python 3.12 environment and
installs Python if needed. No GPU, API keys or separately running database server are required.

The pinned `lance-csr` release provides wheels for these platforms:

| Platform | Requirement |
|---|---|
| Apple Silicon Mac | macOS 11 or later |
| Linux x86_64 | glibc 2.28 or later, such as Ubuntu 20.04+, and an AVX2-capable CPU |
| Linux ARM64 (aarch64) | glibc 2.28 or later, such as Ubuntu 20.04+ |
| Windows x86_64 | 64-bit Windows and an AVX2-capable CPU |

**Windows x86_64 users do not need WSL2.** The setup command below installs the native
Windows wheel for `lance-csr` 0.9.1 without compiling Rust. WSL2 remains an optional
way to use the Linux wheel.

Intel Macs and native Windows ARM64 do not have wheels for this pinned release.
Use a supported machine for the workshop. For other environments, consult the
[lance-csr source-build instructions](https://github.com/hyperdimensionalcomputingai/lance-csr/blob/main/python/README.md#installation)
before the session.

Clone the repository and install the locked dependencies:

```bash
git clone https://github.com/hyperdimensionalcomputingai/cdl-2026.git
cd cdl-2026
uv sync --locked --no-build-package lance-csr
```

The wheel-only option for `lance-csr` prevents an unexpected Rust source build. Linux uses
CPU-only PyTorch wheels to avoid downloading CUDA dependencies. Allow the install to finish
before arriving; the exercises use data included in the repository.

## Open the notebooks

From the repository root, start JupyterLab:

```bash
uv run jupyter lab src/
```

Open the local URL printed in the terminal if your browser doesn't open automatically.
Start with `01_sales_order_graph.ipynb` and select the Python 3 kernel. Run cells from top
to bottom with **Shift+Enter**, pausing to inspect each result. The workshop alternates
between explanations in the presentation and hands-on work in these notebooks.

If you get a missing-package error, stop JupyterLab and relaunch it with the command above
so it uses this project's environment. If you run cells out of order, restart the kernel
and run from the top. Keep the notebooks in `src/`, beside the shared data module.

The notebooks create local `*.lancedb` directories, which Git ignores. Setup cells rebuild
the teaching tables when rerun; use these notebooks with the supplied workshop data.

## Check your environment

For a complete check before the session, execute all four notebooks without opening a browser:

```bash
uv run pytest
```

The tests run in temporary directories and leave the saved notebook outputs unchanged.

## What's underneath

[LanceDB](https://docs.lancedb.com/) stores the tables and hypervectors,
[lance-csr](https://github.com/hyperdimensionalcomputingai/lance-csr) runs Cypher over the
graph, and [torchhd](https://github.com/hyperdimensional-computing/torchhd) supplies the HDC
operators. The shared dataset and helpers are in [src/sales_orders.py](src/sales_orders.py);
workshop figures are in [img/](img/).

See the [CDL talk page](https://2026.connected-data.london/talks/modeling-connected-data-with-graphs-and-hypervectors/)
for the session details. The scenario builds on
[Your data is connected](https://hyperdimensionalcomputing.ai/whats-possible/your-data-is-connected/).
