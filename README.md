### Re-implementation of a GPT model

This repository contains an implementation of GPT-2 model and it's smaller
variants and it's training on a random philosophy article. To install simply
run `pip install -e .` from the cloned repo. In order to keep the progress
updates in the notebook when it's being run for a long time despite
re-connections, modify the `jupyter_notebook_config.py` file by adding
```{python}
c.ServerApp.rate_limit_window = float("inf")
```
To make sure that during training bokeh updates the plot as expected
I had to use jupyterlab instead of jupyter.
