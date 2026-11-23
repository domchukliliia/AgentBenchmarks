# Metódy hodnotenia agentických systémov UI II

Repozitár obsahuje experimentálnu infraštruktúru, skripty a výsledky hodnotenia agentických systémov založených na lokálnych veľkých jazykových modeloch (LLM) pomocou benchmarkov **GAIA** a **τ-bench**.

## Štruktúra repozitára

```text
AgentBenchmarks/
├── README.md                  # Dokumentácia projektu a prehľad výsledkov
├── requirements.txt           # Zoznam závislostí Pythonu
├── .gitignore                 # Ignorované súbory (.venv, data, cache)
├── gaia/
│   ├── scripts/               # Evaluačné skripty pre benchmark GAIA
│   └── results/               # Výsledky experimentov (.json, .jsonl)
└── tau-bench/
    ├── custom_scripts/        # Vlastné skripty pre spúšťanie evaluácií
    ├── results/               # Výsledky behov na doméne Retail (.json)
    └── tau_bench/             # Jadro benchmarku τ-bench