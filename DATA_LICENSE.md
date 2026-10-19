# Source data attribution and reuse

The unmodified file `data/raw/IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv` is redistributed from:

Esmaeel Mohammadi, Anju Rani, Mikkel Stokholm-Bjerregaard, Daniel Ortiz-Arroyo, and Petar Durdevic (2024). *Wastewater Treatment Plant Data for Nutrient Removal System*. Mendeley Data, version 2. DOI: [10.17632/34rpmsxc4z.2](https://doi.org/10.17632/34rpmsxc4z.2).

Authoritative source: <https://data.mendeley.com/datasets/34rpmsxc4z/2>. The source page, checked on 19 October 2026, identifies the licence as **Creative Commons Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)**: <https://creativecommons.org/licenses/by-nc/3.0/>.

Retain the dataset title, creator attribution, DOI, and licence link. The source licence permits sharing and adaptations for noncommercial use under its terms. This repository does not claim ownership of the source archive or endorsement by its creators.

Raw-file SHA-256: `6734fa9e7a351b5823c501d2d297b27923783f93ed7aa1c9c68b8e0ec0143a07` (43,675,886 bytes). The distributed source is already a processed SCADA archive; it is called raw here only relative to this study's further transformations.

Derived data in `data/processed/`, `data/predictions/`, and `tables/` were generated for the present study by 10-minute aggregation, causal feature engineering, chronological splitting, forecasting, calibration, and statistical summarisation. These transformations are documented in the code and do not remove the source data's attribution and noncommercial requirements. Binary `.partNNN` files are exact byte fragments, not numerical transformations.

The source data licence is distinct from the analysis code. No new blanket software licence is assigned by this release; third-party Python dependencies retain their own licences.
