# Operator Learning

## Installation instructions
The package is not yet available on `pypi` — that is, it is not yet installable with `pip`. It will be soon. To develop this package I'm using [`uv`](https://docs.astral.sh/uv/), which is a modern and *way faster* package manager than `conda`. To get up and running, just follow these three steps:

1. Install `uv` following the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone the repository
    ```bash
    git clone git@github.com:pietronvll/operator_learning.git
    ```
3. Open a terminal window in the `operator_learning` folder just cloned, and install the dependencies by running 
    ```bash
    uv sync --dev
    ```

## Development
I use [Visual Studio Code](https://code.visualstudio.com/), which should automatically prompt you to use the virtual environment found in the `operator_learning` folder.

### Adding a dependency
As of now, `operator_learning` only depends on `numpy`, `scipy`, and `torch`. If you implement a functionality needing an additional dependency `dep`, you can add it to the project by simply running
```bash
uv add dep
```
If you need a *development* dependency `dev_dep`, something that you use for testing but is not needed for the core functionality, you can add it in the `--dev` group by running
```bash
uv add --dev dev_dep
```
For example, in the current `--dev` group — which you can read off `pyproject.toml` — we have `scikit-learn`, `matplotlib`, and HuggingFace's `datasets`.

