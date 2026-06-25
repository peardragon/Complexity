---
title: "Deep Expansion Summary"
tags: ["megasearch", "deep-expansion", "citation-chase"]
aliases: ["deep expansion summary"]
created: 2026-06-22
source: "OpenAlex + arXiv + DOI resolver + scholar-megasearch PDF fetch"
confidence: high
---

# Deep Expansion Summary

- date: 2026-06-22
- mode: deepest available fallback mega research + recent citation chase
- primary APIs/tools: OpenAlex, arXiv API, DOI resolver, web verification, scholar-megasearch PDF fetch script
- top prior-art seeds checked for recent citing/adjacent work: 10
- verified references after expansion: 60
- newly added verified references: 25
- paper cards after expansion: 60
- expanded PDF acquisition: 49/60 public PDFs acquired; 11 marked `needs_mcp`

## Added Artifacts

- `05_Claim_Evidence_Matrix.md`: expanded claim/evidence matrix with 16 claim rows and recent-citation interpretation.
- `08_Recent_Citation_Chase.md`: top-prior-art seed table and recent citing/adjacent papers.
- `09_Expanded_Reference_List.md`: 60 verified references.
- `megasearch/openalex_recent_citers.json`: raw OpenAlex recent-citer output for top seeds.
- `megasearch/expanded_verified_references.json`: machine-readable 60-reference list.
- `megasearch/pdfs_expanded/manifest.json`: expanded PDF acquisition status.

## Verification Notes

- Semantic Scholar remained rate-limited in this environment, so recent-citation confirmation used OpenAlex plus DOI/arXiv/official venue checks.
- Broad OpenAlex citing results were filtered because high-citation survey/application papers often cite the seed papers without being directly relevant to local entropy, loss geometry, or shell measurement.
- Citation keys in the final markdown files were checked against `07_Bibliography.bib`; no missing keys remained after the expansion pass.
