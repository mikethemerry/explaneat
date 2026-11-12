### PropNEAT runs (selection by training AUC; test reported only)

| Base | Config | Seed | Train AUC | Test AUC | Hidden | Conns | Inputs used | Best gen | Wall (min) |
|---|---|---|---|---|---|---|---|---|---|
| lines | default | 0 | 0.8723 | 0.8702 | 13 | 52 | 14 | 99 | 19.2 |
| lines | default | 1 | 0.8681 | 0.8665 | 13 | 41 | 10 | 99 | 17.3 |
| lines | default | 2 | 0.8709 | 0.8694 | 9 | 31 | 10 | 99 | 16.3 |
| lines | heavy | 0 | **0.8815** | 0.8785 | 11 | 127 | 16 | 79 | 73.1 |
| lines | heavy | 1 | 0.8815 | 0.8786 | 11 | 109 | 16 | 79 | 70.5 |
| lines | heavy | 2 | 0.8808 | 0.8776 | 10 | 129 | 16 | 79 | 101.4 |
| lines_cells | default | 0 | 0.8862 | 0.8864 | 7 | 60 | 48 | 99 | 23.6 |
| lines_cells | default | 1 | 0.8872 | 0.8885 | 8 | 59 | 42 | 99 | 24.2 |
| lines_cells | default | 2 | 0.8861 | 0.8859 | 4 | 46 | 40 | 99 | 22.7 |
| lines_cells | heavy | 0 | 0.9344 | 0.9289 | 10 | 648 | 58 | 79 | 114.1 |
| lines_cells | heavy | 1 | 0.9337 | 0.9282 | 10 | 648 | 58 | 79 | 126.6 |
| lines_cells | heavy | 2 | **0.9352** | 0.9293 | 10 | 648 | 58 | 79 | 107.6 |
| perline | default | 0 | 0.9115 | 0.9146 | 6 | 98 | 87 | 99 | 37.9 |
| perline | default | 1 | 0.9107 | 0.9137 | 2 | 81 | 79 | 99 | 37.5 |
| perline | default | 2 | **0.9120** | 0.9140 | 3 | 86 | 82 | 99 | 35.1 |

Bold: the genome selected per base (largest training AUC). Per base and config, mean (SD) of test AUC:

- lines / heavy: n = 3, test AUC 0.8782 (SD 0.0006), range 0.8776-0.8786
- lines / default: n = 3, test AUC 0.8687 (SD 0.0019), range 0.8665-0.8702
- lines_cells / heavy: n = 3, test AUC 0.9288 (SD 0.0005), range 0.9282-0.9293
- lines_cells / default: n = 3, test AUC 0.8869 (SD 0.0014), range 0.8859-0.8885
- perline / default: n = 3, test AUC 0.9141 (SD 0.0004), range 0.9137-0.9146

### References on the same split (train / test AUC)

| Base | Inputs | LR | GBM | PropNEAT (earlier studies) |
|---|---|---|---|---|
| cells (earlier) | 42 | 0.857 / 0.861 | 0.947 / 0.944 | complex_real default 0.858 / 0.862; capacity 0.919 / 0.915 (selected), mean test 0.911 |
| lines | 16 | 0.865 / 0.862 | 0.894 / 0.885 | this study (above) |
| lines_cells | 58 | 0.888 / 0.888 | 0.949 / 0.943 | this study (above) |
| perline | 154 | 0.917 / 0.919 | 0.959 / 0.952 | this study (above) |

### Signal check: GBM / LR test AUC by nameable vocabulary (`signal_check.py`)

| Vocabulary | Inputs | LR test | GBM test |
|---|---|---|---|
| aggregate16 | 16 | 0.8625 | 0.8850 |
| direction24 | 24 | 0.8800 | 0.9152 |
| rowband36 | 36 | 0.8848 | 0.9181 |
| aggregate16+direction24+rowband36 | 76 | 0.8999 | 0.9360 |
| perline138 | 138 | 0.9172 | 0.9530 |
| perline138+aggregate16 | 154 | 0.9188 | 0.9524 |
| cells42 | 42 | 0.8611 | 0.9437 |
