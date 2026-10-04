# NIRLab

A Streamlit app for exploring near-infrared (NIR) spectra exported as multi-block
[JCAMP-DX](http://www.jcamp-dx.org/) (`.jdx`) files. It takes a raw export through a standard
chemometrics workflow, from raw spectra to PLS regression, without writing code.

The analysis logic lives in `nir_core.py`, independent of the interface, so it can be reused
from scripts or another front end. The interface text is in Italian.

## Workflow

The app has one tab per step:

1. **Raw spectra** — load `.jdx` files; each block becomes one sample, exported as a
   wavelength × sample matrix.
2. **L, a, b** — CIE Lab colour values stored with each block (`##CONCENTRATIONS=`), if present.
3. **Outliers** — distance of each SNV spectrum from the mean, to spot suspicious samples.
4. **Duplicates** — reproducibility of repeated measurements (an original and a `bis` replicate).
5. **SNV** — Standard Normal Variate correction on the final dataset, after excluding samples
   and averaging duplicates.
6. **PCA** — scores and loadings of the SNV spectra, with points optionally coloured by L, a or b.
7. **Clustering** — Ward hierarchical clustering with dendrogram and silhouette score.
8. **PLS** — regression of the SNV spectra on a chosen target variable, with leave-one-out
   cross-validation to pick the number of components.

Tables can be downloaded as CSV from each tab.

## Run

```
pip install -r requirements.txt
streamlit run app_streamlit.py
```

On Windows, `Avvia_App.bat` starts the app with `py -m streamlit run app_streamlit.py`.

## Data organisation

Code and data are kept apart. Each analysis is a **project** under `data/`, created from the
sidebar:

```
data/<project>/
├── raw/            source .jdx files
├── <group>/        CSVs and figures for each sample group
├── LAB_values.csv  files shared across groups
└── reports/        write-ups
```

Samples are grouped by the alphabetic prefix of their name (`Sal12` → `sal`), and a trailing
`bis` marks a duplicate of the sample with the same number (`Sal7bis`).

`data/` is excluded from version control: spectra and results stay on your machine.

## Input format

The parser expects multi-block JCAMP-DX files where each block has a `##TITLE=`, an
`##XYPOINTS=(XY..XY)` section and, optionally, `##CONCENTRATIONS=` with `L`, `a`, `b` values.
It was developed on exports from a BUCHI NIRWise instrument (diffuse reflectance, 400–1700 nm).
Files from other instruments may need small changes to the parser.

## Requirements

Python 3.10+ and the packages in `requirements.txt` (streamlit, pandas, numpy, scipy,
scikit-learn, matplotlib, plotly).
