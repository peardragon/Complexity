# Megasearch Summary

- depth: L5 Total/exhaustive fallback
- topic: local entropy, Franz-Parisi potential, shell sampling, neural-network solution geometry, dataset complexity
- raw source count status: {'ok': 43, 'failed': 8, 'empty': 20}
- unique corpus: 405
- min-source>=2 shortlist: 16
- source note: scholar-megasearch was cloned and its merge/fetch scripts were used. Full MCP fan-out was unavailable in this session, so raw acquisition used local fallback APIs: arXiv, OpenAlex, Crossref, DBLP, DDG, PubMed, and partial Semantic Scholar. Semantic Scholar rate-limited several facets with HTTP 429; this is recorded in `manifest.json`.

## Query Facets

See `query_plan.json`.

## Recommended Use

Use `corpus_shortlist_min2.*` as high-precision discovery evidence, but use `07_Bibliography.bib` for manuscript citations because it has been manually curated for DOI/arXiv/venue/year hygiene.
