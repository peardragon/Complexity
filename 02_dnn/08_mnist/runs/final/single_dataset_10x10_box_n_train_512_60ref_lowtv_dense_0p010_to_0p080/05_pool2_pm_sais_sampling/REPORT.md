# Stage 05 PM-SAIS Sampling

Completed 840 shell units with 0 fallback-policy units.

Sample counts present: [4096]

Fallback total-sample counts: {}

Sampling note: Low-TV-only dense extension. Reuses identical 10x10 BOX dataset, exact references, and already completed shell units where available; computes missing radii with the unchanged PM-SAIS unit sampler.

All selected rule/radius rows passed the Stage 05 QC gates.

## Runtime

Pilot timing used `max_units=14`, covering one reference across all radii; 5 newly computed units took 22.02 s, estimating roughly 22.0 min for 300 missing units before reuse and summary overhead.

Actual Stage 05 elapsed time was 740.94 s for 840 unit rows: 540 copied/reused unit summaries and 300 newly computed unit summaries.
