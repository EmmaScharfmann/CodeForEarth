# CodeForEarth

This repository provides a python package for identifying **weather regimes** insightful about energy consumption from climate data, using a custom **Conditional Mixture-Model VAE
(CMM-VAE)**. The method is based on this repository: https://github.com/fiona511/predictability_paper. The goal of the method is to define regimes which are informative about the energy consumption. The package also provides a classical PCA + K-means baseline,
evaluation metrics (Brier skill score, ROC AUC), cartopy-based plotting utilities, and tools to
score S2S (subseasonal-to-seasonal) forecasts against these regimes.

It was built as part of a CodeForEarth project, organized by ECWMF, by Emma Scharfmann, Quentin
Nicolas, Nora Zilibotti, and Vishnupriya Selvakumar.

> **Note on naming:** the importable package is currently called `package_name` (i.e. you write
> `from package_name.training.trainer import VAETrainer`, not `from codeforearth import ...`).
> This is a placeholder left over from the project template — see [Limitations](#limitations--discussion).

---

## 1. How to use the package

### 1.1 Setup

```bash
# Clone the repo
git clone https://github.com/EmmaScharfmann/CodeForEarth.git
cd CodeForEarth

# Install the package in editable mode
pip install -e .

# Install the remaining dependencies (xesmf/esmpy are easiest to get via conda-forge)
conda install -c conda-forge esmpy
pip install -r requirements.txt
```

Requirements: Python ≥ 3.13. The stack is TensorFlow/Keras for the model, xarray/netCDF4/xesmf
for climate-data handling, scikit-learn for the baseline and metrics, and cartopy/matplotlib/
seaborn for plotting.

### 1.2 A highlevel overview of the method

1. You have a **predictor field** `X` (e.g. daily Z500 geopotential height anomalies over a
   region), shaped `(n_times, n_lat, n_lon)`.
2. You have a **target label** `y` you want the regimes to be informative about (e.g. a
   one-hot/soft encoding of which energy cluster each day falls into), shaped
   `(n_times, n_target_classes)`.
3. You configure and train a `VAETrainer`, which fits a probabilistic encoder that maps `X` to a
   latent space organized around `cluster_number` Gaussian components, jointly with a classifier
   head that ties clusters to the target `y`.
4. You load the trained weights into a `VAEPredictor` to get per-day cluster probabilities
   (regime assignments), decode cluster centers back into physical space, and score how
   informative the regimes are about the target with Brier skill scores.

### 1.3 End-to-end walkthrough

```python
import numpy as np
from package_name.model.utils import sampling
from package_name.training.trainer import VAETrainer, VAEConfig
from package_name.inference.predictor import VAEPredictor
from package_name.data_processing.data_processor import flatten_input
from package_name.evaluation import metrics
from package_name.evaluation.utils import predict_clusters

# --- 1. Data -------------------------------------------------------------
# X: (n_times, n_lat, n_lon) predictor field, e.g. Z500 anomalies
# y: (n_times, pr_cluster_number) target encoding, e.g. precipitation-regime membership
X = ...  # np.ndarray
y = ...  # np.ndarray
n_times, n_lat, n_lon = X.shape

# --- 2. Configure and train the CMM-VAE -----------------------------------
config = VAEConfig(
    original_dim=n_lat * n_lon,     # flattened spatial dimension
    original_dim_target=y.shape[1],
    dim_layer1=256,
    dim_layer2=128,
    dim_layer3=64,
    activation="relu",
    cluster_number=7,               # number of weather regimes to fit
    latent_dim=15,
    pr_cluster_number=y.shape[1],
    sampling_fn=sampling,
)

vae = VAETrainer(cfg=config, path_to_save_weights="results/")
vae.compile()
history = vae.fit(X=X, y=y, epochs=70, batch_size=128)

vae.save_weights("results/final_weights.weights.h5")

# --- 3. Run inference ------------------------------------------------------
predictor = VAEPredictor(cfg=config)
predictor.load_weights("results/final_weights.weights.h5")

cluster_probabilities = predict_clusters(vae=predictor, inputs=X)   # (n_times, cluster_number)
cluster_labels = cluster_probabilities.argmax(axis=1)

# --- 4. Evaluate how informative the regimes are ---------------------------
bss = metrics.compute_BSS_clusters_target(vae=predictor, inputs=X, targets_categorical=y)
print(f"Brier skill score (regimes -> target): {bss:.3f}")
```

You can find a full working example, including loading ERA5 Z500 and CHIRPS precipitation data, cross-
validation, plotting, and comparison against the PCA + K-means baseline — is in
`experiments/cmmvae.ipynb` and an S2S-forecast scoring example is in `experiments/cmmvae_s2s.ipynb`.

---

## 2. What the package is about

This package implements a **Categorical/Gaussian Mixture VAE** that is learns:

- a latent space `z`, sampled from an encoder `q(z | x)` (the usual VAE reparameterization trick,
  see `package_name/model/utils.py::sampling`),
- a **Gaussian mixture prior** over that latent space, with learned component means `mu` and
  mixing weights `pi` (`package_name/model/encoder.py::_build_mixture_components`) — these
  mixture components *are* the weather regimes,
- a **cluster-assignment head** that predicts, from `x` alone, the probability of belonging to
  each mixture component,
- a **target-prediction head** that predicts the impact variable `y` from the same features used
  for cluster assignment, and is regularized to agree with the cluster assignment.

The loss is made of five terms: 
- VAE reconstruction, a regularization term pulling `z` towards its assigned mixture component,
- a KL term between the predicted target distribution and the true target, 
- a consistency term between cluster predictions
made from `x` and from the target
- a Dirichlet-style regularizer on `pi` to discourage empty
clusters.
In other words, the reconstruction and mixture-prior terms push the model to find
circulation-based regimes, while the target-prediction and consistency terms push those regimes to
be predictive of the chosen impact variable.
---

## 3. What is possible with the package

- **Train a CMM-VAE weather-regime model** on any gridded predictor field and target label
  (`package_name/training/trainer.py::VAETrainer`) — the field and target don't have to be Z500 and
  precipitation or energy.
- **Run inference** with a trained model: encode new data to cluster probabilities and latent
  vectors, decode arbitrary latent points (e.g. mixture means) back to physical space, and inspect
  the learned mixture components (`package_name/inference/predictor.py::VAEPredictor`).
- **Cross-validate** by retraining the same architecture from the same random initial weights
  multiple times and collecting the training histories, to gauge sensitivity to random
  initialization (`package_name/tuning/cross_validator.py::CrossValidator`).
- **Score how informative regimes are about a target variable** via the Brier skill score,
  computed either directly against the training target, against exceedance of an arbitrary
  quantile threshold, or against quantile/tercile bins of a continuous variable
  (`package_name/evaluation/metrics.py`).
- **Fit and use a classical PCA + K-means baseline** with the same probabilistic-cluster-output
  interface, for comparison against the CMM-VAE (`package_name/evaluation/pca_kmeans.py`).
- **Visualize regimes on a map**: decoded cluster centers, empirical (composite) cluster centers,
  and the spatial odds ratio of a binary impact variable within each cluster, all as cartopy
  contour maps (`package_name/evaluation/plots.py`).
- **Evaluate S2S (subseasonal-to-seasonal) forecasts** against the fitted regimes: bootstrap Brier
  skill score and multiclass ROC AUC by forecast lead time, with plotting
  (`package_name/evaluation/s2s_eval.py`).
- **Preprocess gridded climate data** (region selection over several predefined domains, seasonal
  filtering, anomaly computation, normalization, rolling-mean smoothing, and regridding) via the
  helpers in `experiments/utils.py`.

### Package map

| Module | Responsibility |
|---|---|
| `data_processing/data_processor.py` | Flatten/unflatten spatial arrays, train/val split, build `tf.data.Dataset`s |
| `model/encoder.py`, `model/decoder.py` | Dense encoder/decoder network builders |
| `model/loss.py`, `model/utils.py` | The composite VAE + mixture + target loss, and config/dataclasses |
| `model/vae_model.py` | The Keras `Model` wiring encoder, decoder, and loss together |
| `training/trainer.py` | `VAETrainer`: build, compile, fit, save weights |
| `inference/predictor.py` | `VAEPredictor`: load weights, encode/decode, inspect mixture components |
| `tuning/cross_validator.py` | Repeated-training cross-validation harness |
| `evaluation/metrics.py` | Brier skill score variants for cluster informativeness |
| `evaluation/pca_kmeans.py` | PCA + K-means baseline with a probabilistic-cluster API |
| `evaluation/plots.py` | Cartopy map plots of cluster centers and odds ratios |
| `evaluation/s2s_eval.py` | Bootstrapped BSS / ROC AUC by lead time for forecast evaluation |
| `experiments/utils.py` | Data loading/preprocessing helpers used by the notebooks (not part of the installed package) |

---

## 4. Limitations & discussion

- **Placeholder package name.** The distribution is `package_name` (see `pyproject.toml`), so
  every import is `from package_name... import ...`. This should eventually be renamed to
  something like `codeforearth` for a public release — until then, don't expect the import path
  to match the repo name.
- **Supervised, not purely unsupervised, clustering.** Unlike classical weather-regime methods,
  the CMM-VAE *requires* a target label `y` at training time to shape the clusters. This is the
  point of the method (impact-aware regimes), but it also means the regimes you get are informative 
  about this variable. 
- **No spatial structure in the network.** The encoder/decoder are plain dense (MLP) stacks over a
  flattened `(lat × lon)` vector (`data_processor.flatten_input`); there is no convolutional or
  graph structure exploiting spatial locality. This is simple and fast for the modest grids used
  here, but likely won't scale gracefully to high-resolution fields.
- **A few loss-weighting choices are empirical.** For example, the `VAELoss` multiplies the
  reconstruction loss by `original_dim` and the target-prediction loss by `pr_cluster_number`.
- **Cross-validation is a repeated-training harness, not a statistical cross-validation.**
  `CrossValidator` retrains the same model `n_runs` times and returns the raw training histories;
  it does not (yet) compute evaluation metrics per run or a confidence interval. This is
  explicitly marked as a TODO.
- **`get_mixture_components` relies on a documented workaround.** Because the mixture parameters
  `mu`/`pi` only depend on a dummy constant input, they're read out by feeding the encoder a
  dummy batch of size 1 rather than through a dedicated code path (see the `TODO` in
  `inference/predictor.py`).
- **Heavy, geoscience-specific dependency stack.** TensorFlow/Keras, xarray, netCDF4, and
  xesmf/esmpy (which is only reliably installed via conda) make the environment nontrivial to set
  up, especially outside conda-based workflows, and pin the project to an unusually recent Python
  (≥ 3.13).
- **No automated test suite.** There are no `test_*.py` files in the repository despite the
  documented testing conventions in this README; correctness currently relies on manual notebook
  runs (`experiments/cmmvae.ipynb`, `experiments/cmmvae_s2s.ipynb`) rather than `pytest`.
---

## Development

```bash
black .    # format the code before opening a PR
```

See the docstring, naming, and formatting conventions the project follows in
`CONTRIBUTING`-style form below:

- Public functions/methods get a full docstring (one-line summary + inputs/outputs); private
  functions may have just a one-line description. Comments are reserved for non-obvious choices,
  not for restating what the docstring already says.
- `snake_case` names, function names start with a verb (`build_model`, `get_data`, ...), type
  hints are as precise as possible (prefer small dataclasses over loose dicts), and empty
  container initializations are annotated (`new_list: list[int] = []`).
- Run `black .` before creating a PR.

## License

MIT — see `LICENSE`.
