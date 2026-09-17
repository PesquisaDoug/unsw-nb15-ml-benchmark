# Benchmarking Supervised and Unsupervised Machine Learning for Network Intrusion Detection on UNSW-NB15

This repository modularizes the original executed notebook into a reproducible Python project for a classical ML benchmark on the official predefined UNSW-NB15 train/test partition.

## Research Questions

- RQ1 - Supervised performance: How do Random Forest, XGBoost, Support Vector Machine and Multilayer Perceptron compare for binary network intrusion detection?
- RQ2 - Attack-wise detection: Do aggregate metrics hide important differences in detection performance across attack categories?
- RQ3 - Unsupervised structure: Can k-Means and DBSCAN reveal traffic structure related to normal and malicious behavior without using target labels during fitting?
- RQ4 - Performance/efficiency trade-off: What trade-offs exist between predictive performance, model size, training cost and inference cost?

## Dataset

Dataset: UNSW-NB15.

Canonical dataset page:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

The benchmark preserves the official predefined CSV partition:

- training: `UNSW_NB15_training-set.csv`, expected 175,341 records
- testing: `UNSW_NB15_testing-set.csv`, expected 82,332 records

The target is binary `label`, with `0 = normal` and `1 = attack`. The `attack_cat` column is metadata for post-hoc attack-wise analysis only. It is not used as a feature for preprocessing, tuning, training, inference, clustering, or permutation importance. The `id` column is also removed from features when present.

## Models

Supervised models:

- Dummy baseline
- Random Forest
- XGBoost
- LinearSVC
- MLPClassifier

LinearSVC is used instead of a kernel SVM because the training set has approximately 175 thousand observations and kernel SVM would be substantially more expensive computationally.

Unsupervised models:

- k-Means
- DBSCAN

## Metrics and Outputs

Classification metrics include accuracy, balanced accuracy, precision, recall/detection rate, F1, ROC-AUC, PR-AUC, MCC, and false positive rate. Bootstrap confidence intervals are computed by resampling fixed official-test predictions; models are not retrained during bootstrap.

Clustering metrics include silhouette, Davies-Bouldin, Calinski-Harabasz, Adjusted Rand Index, and Normalized Mutual Information. ARI/NMI are post-hoc only and do not select DBSCAN parameters.

Main outputs are written under `results/`, figures under `figures/`, and models under `artifacts/models/`.

## Repository Structure

```text
configs/      Experiment and model search configuration
src/          Reusable benchmark modules
scripts/      CLI entry points
app/          Streamlit benchmark explorer
tests/        Fast smoke/unit tests
notebooks/    Reference and result-oriented notebooks
paper/        LaTeX manuscript skeleton
data/         Data manifest and raw data location
```

## Installation

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
# fallback if 3.11 is unavailable:
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

Linux, WSL, or macOS:

```bash
python3.11 -m venv .venv
# fallback if 3.11 is unavailable:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

## QUICK Run

```bash
python -m pytest -q
python -m scripts.run_supervised --config configs/experiment.yaml --profile quick
python -m scripts.run_clustering --config configs/experiment.yaml --profile quick
python -m scripts.generate_results --config configs/experiment.yaml --profile quick
streamlit run app/streamlit_app.py
```

## FULL Run

```bash
python -m scripts.run_supervised --config configs/experiment.yaml --profile full
python -m scripts.run_clustering --config configs/experiment.yaml --profile full
python -m scripts.generate_results --config configs/experiment.yaml --profile full
```

The FULL run should be used only after QUICK completes successfully and enough local CPU/RAM/time is available.

## Notebooks

The original notebook is preserved as `notebooks/00_reference_monolithic.ipynb`. The modular notebooks are lightweight and should load results produced by scripts rather than repeating heavy tuning by default.

## Streamlit

The app is a research demonstrator over processed UNSW-NB15 flow features:

```bash
streamlit run app/streamlit_app.py
```

It does not parse PCAP files, capture packets, ingest raw NetFlow/Syslog, monitor live traffic, or claim to be a production real-time IDS. Inference expects a CSV containing the exact processed feature schema saved in `results/feature_schema.json`.

## Limitations

- UNSW-NB15 was produced in a controlled cyber-range.
- Results on UNSW-NB15 do not demonstrate performance on real operational networks.
- The main task is binary attack detection.
- `attack_cat` is post-hoc metadata, not a feature.
- Hyperparameter search is deliberately limited.
- The unsupervised experiment is exploratory.
- PCA and clustering run on a bounded training-set sample.
- DBSCAN is sensitive to representation, dimensionality, and density parameters.
- No raw PCAP processing or live monitoring is implemented.
- No SCADA, thermoelectric, or Operational Technology validation is claimed.

## Citation

The project should cite the UNSW-NB15 dataset page and associated publications requested by the dataset authors, beginning with:

Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). Military Communications and Information Systems Conference (MilCIS).
