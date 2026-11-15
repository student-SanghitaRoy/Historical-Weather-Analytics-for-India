# Q&A evaluation: Chennai, Cherrapunji, Delhi, Guwahati, Jaisalmer, Leh

Data: ERA5 reanalysis via Open-Meteo, cached in C:\Users\DELL\Downloads\weather_project_step6_report (1)\data_cache\india (1991-01-01 to 2025-12-31)

The answer key is computed independently with plain pandas (not with the engine's own statistics code).

| Metric | Value |
|---|---|
| Questions | 270 (168 answerable, 102 that must be refused) |
| Answerable questions answered | 100.0% |
| Numerical accuracy (of those answered) | 100.0% |
| Source and date-range consistency | 100.0% |
| Correct abstention on must-refuse questions | 100.0% |
| False answers (should have refused) | 0.0% |
| Unsupported-claim rate | 0.0% |
| Overall pass rate | 100.0% |

## By category

| Category | Passed | Total |
|---|---|---|
| average temperature | 24 | 24 |
| baseline comparison | 6 | 6 |
| dry spell | 24 | 24 |
| insufficient data | 6 | 6 |
| missing data | 24 | 24 |
| missing data (gappy) | 6 | 6 |
| month across years | 6 | 6 |
| partial coverage | 6 | 6 |
| refuse: cause | 12 | 12 |
| refuse: compare locations | 6 | 6 |
| refuse: conflicting request | 6 | 6 |
| refuse: forecast | 18 | 18 |
| refuse: nonsense | 18 | 18 |
| refuse: not loaded | 6 | 6 |
| refuse: other location | 6 | 6 |
| refuse: outside period | 12 | 12 |
| refuse: trend | 12 | 12 |
| total rainfall | 24 | 24 |
| wet days | 24 | 24 |
| wet days (monsoon) | 24 | 24 |

Limits: questions are generated from templates and the parser is rule-based, so this measures correctness on the supported question forms, not general language understanding.