## 0.4.0 (2026-02-08)

### Feat

- **loaders**: implement langchain community loaders which use strategy pattern to load correct file ext loader, parse into docs, which is called by dir loader, called by pipeline
- **dir**: restructure around src standard package
- **downloading**: add retries and backoff, functionalize, cleanup

## 0.3.0 (2026-02-06)

### Feat

- **metadata**: capture additional metadata

## 0.2.0 (2026-02-05)

### Feat

- **data-pull**: add script which downloads all files listed in source csv file, with attractive console display

### Fix

- **raw-data**: document data sources
- **pytest**: remove pytest from ci to prevent build error
