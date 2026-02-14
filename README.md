# GEDI_DA
# HIGHER Pipeline Repository

This repository is organized into modular folders. Each folder corresponds to a specific stage of the pipeline and contains its own dedicated README explaining the scripts within that stage in detail.

The structure reflects the full workflow: raw footprint ingestion → standardization and feature engineering → modeling and paper output generation.

---

## Repository Structure

### `GEE_Extract/`

Scripts in this folder are designed to run inside **Google Earth Engine** infrastructure.

They are not runnable inside the current Docker environment because they depend on Google’s internal Earth Engine execution system, authentication stack, and distributed processing backend.

These scripts perform:

* Raster extraction at scale
* Earth Engine asset interaction
* Cloud-side processing

Execution requires:

* An authenticated Google Earth Engine environment
* Access to the relevant Earth Engine assets

They are not intended to run locally.

---

### `rbfp/`

Raw Building Footprints

These scripts ingest heterogeneous building footprint datasets and convert them into a standardized format.

Purpose:

* Harmonize geometry structure
* Normalize CRS
* Standardize schema
* Prepare datasets listed in the `data_link_text/` folder

This stage ensures all footprint sources conform to a consistent structure before feature engineering.

Outputs from this stage are the canonical building-level geometry inputs used downstream.

---

### `ds_mf/`

Data Standardization and Morphological Features

This stage performs:

1. Raster feature extraction
2. Planimetric morphological feature generation
3. Schema standardization across all cities
4. Feature validation and cleaning

Core responsibilities:

* Compute footprint-derived morphology metrics
* Extract raster-based features
* Ensure column consistency across datasets
* Prepare modeling-ready GeoPackages

The output of this stage forms the unified analytical dataset used for model training and evaluation.

---

### `model_scripts/`

This folder contains the scripts that generate the outputs used in the research paper.

Contents include:

* Model training pipelines
* Domain transfer experiments
* Adaptation procedures
* Evaluation metrics
* Plot generation
* Configuration file required to execute experiments

The `config.yaml` file governs:

* Feature selection
* Hyperparameter search
* Adaptation rounds
* Input/output paths

Running the scripts in this folder produces:

* City-level metrics
* Cross-domain evaluation results
* Figures used in publication
* Model artifacts

---

## Execution Order

The pipeline is structured sequentially:

1. `rbfp/` → Standardize raw building footprints
2. `GEE_Extract/` → Extract raster features via Earth Engine
3. `ds_mf/` → Compute morphology + unify schema
4. `model_scripts/` → Train models and generate paper outputs

Each folder README provides implementation-specific details.

---

## Notes

* `GEE_Extract/` requires Google Earth Engine infrastructure.
* Intermediate outputs between stages are required dependencies.
* Scripts are modular; individual folders can be adapted or extended independently.

This repository reflects a full reproducible pipeline from heterogeneous raw building footprints to final modeling outputs.

