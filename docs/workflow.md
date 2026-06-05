# Workflow

```text
SpectroCube files
        │
        ▼
select shot/frame/wavelength windows
        │
        ▼
fit Fulcher-alpha line and blend groups
        │
        ▼
write intensity/error CSVs and QC summaries
        │
        ▼
fulcheranalyzer
```

The first implementation target is the LHD 20250926 SpectroCube batch exported
with absolute radiance units and a 403 nm low-wavelength crop.

