# FINAL TEMPORAL GEOMETRY ANALYSIS

## 1. Artifact Validity & Exact Methods
- **A0**: Valid seeds: [42]
- **A1**: Valid seeds: [42, 123, 2025, 7, 13]
- **A3**: Valid seeds: [42]
- **SoftCLT**: Valid seeds: [42, 123, 2025, 7, 13]

## 2. Sample Details
- Test Subjects: 23
- Total Windows: 19357
- Same-Subject Pairs: 500000
- Cross-Subject Pairs (Control): 500000

## 3. Aggregate Statistical Results
| Method | Seed | Spearman ρ | 95% CI (Patient Boot) | Adj Dist | Distant Dist | Diff | Diff 95% CI | Cross-Sub Dist |
|---|---|---|---|---|---|---|---|---|
| A0 | 42 | 0.165 | [0.137, 0.206] | 1.797 | 4.050 | 2.253 | [1.740, 2.757] | 9.498 |
| A1 | 42 | 0.121 | [0.066, 0.191] | 1.074 | 2.546 | 1.472 | [1.141, 1.796] | 6.084 |
| A1 | 123 | 0.123 | [0.066, 0.201] | 1.439 | 3.550 | 2.110 | [1.494, 2.738] | 6.340 |
| A1 | 2025 | 0.108 | [0.042, 0.188] | 1.088 | 2.747 | 1.659 | [1.126, 2.167] | 7.004 |
| A1 | 7 | 0.124 | [0.070, 0.185] | 1.328 | 3.327 | 1.999 | [1.432, 2.570] | 8.185 |
| A1 | 13 | 0.125 | [0.080, 0.187] | 1.176 | 2.891 | 1.716 | [1.202, 2.766] | 11.074 |
| A3 | 42 | 0.189 | [0.158, 0.234] | 0.716 | 1.322 | 0.606 | [0.443, 0.817] | 2.780 |
| SoftCLT | 42 | 0.126 | [0.074, 0.176] | 2.777 | 6.541 | 3.764 | [3.032, 4.343] | 22.142 |
| SoftCLT | 123 | 0.127 | [0.073, 0.180] | 1.058 | 2.366 | 1.308 | [1.013, 1.598] | 17.749 |
| SoftCLT | 2025 | 0.139 | [0.041, 0.216] | 0.461 | 1.079 | 0.618 | [0.464, 0.737] | 4.052 |
| SoftCLT | 7 | 0.168 | [0.115, 0.222] | 0.527 | 1.238 | 0.711 | [0.573, 0.822] | 4.898 |
| SoftCLT | 13 | 0.159 | [0.128, 0.207] | 1.286 | 3.453 | 2.167 | [1.552, 2.702] | 18.813 |

## 4. Scientific Answers
**Q1. Does A1 produce a stronger temporal-distance/representation-distance relationship than A0?**
No. The correlation for A1 (mean 0.120) is not meaningfully stronger than A0 (0.165).

**Q2. Are adjacent same-subject windows measurably closer in A1?**
Yes, across all valid seeds, adjacent windows are consistently closer than distant windows.

**Q3. Is this effect reproducible across seeds?**
A1 was evaluated on 5 valid seeds. The variance in correlation was 0.006.

**Q4. Does A3 exhibit a similar effect?**
Yes, A3 shows a correlation of 0.189.

**Q5. Does SoftCLT exhibit a similar effect?**
SoftCLT shows a correlation of 0.144.

**Q6. Is the effect specific to A1, or do all temporal methods show it?**
See above correlations. If SoftCLT or A3 achieve similar/stronger temporal structuring, the effect is not specifically unique to A1's kernel mechanism.

**Q7. Does the analysis provide legitimate evidence that A1's mathematical mechanism actually changes representation geometry in the intended temporal direction?**
No, there is no robust evidence that the representation geometry meaningfully reflects the intended temporal structure more than the vanilla baseline.

**Q8. Does this justify a mechanism-level claim in the paper?**
No. The mechanism-level claim must be removed or heavily qualified.


### VERDICT

**MECHANISM NOT SUPPORTED**

1. We computed patient-level bootstrap CIs over hundreds of thousands of pairs to ensure rigor.
2. The analysis evaluates exact valid checkpoints to avoid seed-collapse artifacts.
3. The Spearman correlation evaluates monotonic temporal structure rather than simple absolute distances.
