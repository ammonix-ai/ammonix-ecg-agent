# Third-party notices

## Fonts

- **Hanken Grotesk** and **IBM Plex Mono**, the viewer's fonts, are SIL Open Font License 1.1.
  They are installed from npm (`@fontsource-variable/hanken-grotesk`, `@fontsource/ibm-plex-mono`)
  and bundled into the built viewer, with their license texts inside those packages.
- **Inter** (`frontend/public/fonts/InterVariable.woff2`) and **JetBrains Mono**
  (`frontend/public/fonts/JetBrainsMonoVariable.woff2`) are shipped but no longer used by the
  viewer. SIL Open Font License 1.1, texts in `frontend/public/fonts/Inter-LICENSE.txt` and
  `frontend/public/fonts/JetBrainsMono-LICENSE.txt`.

## Software dependencies

Installed from their package registries, not redistributed here. Python: FastAPI, Uvicorn,
Pydantic, NumPy, SciPy, scikit-learn, XGBoost, wfdb, httpx, python-multipart. JavaScript: see
`frontend/package.json`. Rust: see `qpsi/qpsi_native/Cargo.toml`. All are under permissive
licenses (MIT, BSD, Apache-2.0, ISC, MPL-2.0).

## Datasets

No dataset is redistributed in this repository. The software was developed and evaluated on
the following public research datasets, each under its own terms. If you use them, cite them
and follow their terms.

- P. Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset,"
  *Scientific Data* 7, 154 (2020). CC BY 4.0.
- F. Liu et al., "An open access database for evaluating the algorithms of electrocardiogram
  rhythm and morphology abnormality detection," *J. Med. Imaging Health Inform.* 8(7), 2018
  (CPSC 2018).
- J. Zheng et al., "A 12-lead electrocardiogram database for arrhythmia research covering more
  than 10,000 patients," *Scientific Data* 7, 48 (2020) (Chapman-Shaoxing).
- E. A. Perez Alday et al., "Classification of 12-lead ECGs: The PhysioNet/Computing in
  Cardiology Challenge 2020," *Physiol. Meas.* 41, 124003 (2020) (includes the Georgia cohort).
  CC BY 4.0.
- B. Gow et al., "MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset," PhysioNet. Open
  Database License 1.0.
- N. Costa Cortez and D. Garcia Iglesias, "Brugada-HUCA: 12-Lead ECG Recordings for the Study
  of Brugada Syndrome," version 1.0.0, PhysioNet, 2026, doi:10.13026/0m2w-dy83. CC BY-SA 4.0.
- V. Asthana et al., "Pre-/Post-STEMI ECG Database," University of Michigan, Deep Blue Data,
  2024, doi:10.7302/gk9v-ka27. Terms as stated on the Deep Blue record.
- A. Johnson et al., "MIMIC-IV-Note: Deidentified free-text clinical notes," version 2.2,
  PhysioNet, 2023, doi:10.13026/1n74-ne17. Credentialed access (PhysioNet Credentialed Health
  Data License 1.5.0). Used by the authors for some training labels; nothing derived from it is
  in this repository.
- A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation* 101(23),
  2000.
