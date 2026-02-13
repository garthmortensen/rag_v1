## 0.7.0 (2026-02-12)

### Feat

- **retrieval**: implement retrieval and tests

## 0.6.0 (2026-02-12)

### Feat

- **embeddings**: implement embeddings and model selection based on VRAM

## 0.5.0 (2026-02-11)

### Feat

- **embeddings**: add starter embeddings feature
- **chunking**: implement chunking using langchain text_splitters
- **loaders**: implement langchain community loaders which use strategy pattern to load correct file ext loader, parse into docs, which is called by dir loader, called by pipeline
- **dir**: restructure around src standard package
- **downloading**: add retries and backoff, functionalize, cleanup
- **metadata**: capture additional metadata
- **data-pull**: add script which downloads all files listed in source csv file, with attractive console display

### Fix

- **raw-data**: document data sources
- **pytest**: remove pytest from ci to prevent build error
