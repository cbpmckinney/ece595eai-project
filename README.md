# ECE 595 – Exploration  of Implicit Bias Towards Minor Users in LLM Responses

This repository contains the code, datasets, and analysis scripts for an experimental evaluation of **gender bias in large language model (LLM) recommendation behavior for children and adolescents**.

The project systematically probes two LLMs using a fully factorial prompt design and records both initial recommendations and follow-up justifications (“Why?”) for subsequent human coding and analysis.

---

## Project Overview

We evaluate whether LLMs exhibit gender-stereotyped behavior when asked to provide age-appropriate recommendations to children.

### Key experimental dimensions:
- **LLMs under test:** 2 (Claude, Llama-4)
- **Model role framing:** 2 (“helpful advisor”, “educator”)
- **User gender labels:** 3 (“boy”, “girl”, “child”)
- **User ages:** 13 (ages 3–15)
- **Recommendation domains:** 4 (toys, hobbies, careers, academic subjects)

Each prompt is followed by a standardized second-turn probe (“Why?”) to elicit the model’s underlying reasoning.

---

## Experimental Procedure

For each experimental condition:

1. A prompt is generated using a fill-in-the-blank template.
2. The prompt is sent to the LLM.
3. The model’s response is recorded.
4. A follow-up prompt (“Why?”) is sent.
5. The justification response is recorded.
6. The conversation context is reset before the next trial.

Follow-up scripts analyze the body of results using cosine distance, PCA, lexical analysis, and Latent Dirichlet Allocation.

---

## Data Generation

Raw prompts are generated using a factorial design covering all combinations of:
- model × role × gender × age × domain

Each LLM is queried independently using finalized query scripts.

---

## Data Cleaning

Minimal post-processing is applied to standardize formatting across model outputs.

Specifically:
- Boilerplate text preceding numbered recommendation lists is removed.
- Responses are otherwise preserved verbatim.

This cleaning step does **not** alter semantic content and is performed solely to improve consistency during human review and analysis.

---

## Repository Structure

├── README.md<br>
│<br>
├── data/<br>
│ ├── gender_bias_tests.json<br>
│ │ Generated prompt dataset (all factorial combinations)<br>
│<br>
│ ├── claude3results.json<br>
│ │ Raw Claude responses<br>
│ ├── llama4results.json<br>
│ │ Raw Llama-4 responses<br>
│<br>
│ ├── claude3results-sanitized.json<br>
│ │ Cleaned Claude responses<br>
│ └── llama4results-sanitized.json<br>
│ Cleaned Llama-4 responses<br>
│<br>
├── query-claude.py<br>
│ Sends prompts and follow-up queries to Claude and records responses<br>
│<br>
├── query-llama.py<br>
│ Sends prompts and follow-up queries to Purdue-hosted Llama-4 and records responses<br>
│<br>
├── sanitize-claude.py<br>
│ Applies light formatting cleanup to Claude outputs<br>
│<br>
├── sanitize-llama.py<br>
│ Applies light formatting cleanup to Llama-4 outputs<br>
│<br>
├── analyze-claude.py<br>
│ Analysis utilities for Claude results (descriptive statistics, grouping)<br>
│<br>
├── analyze-llama.py<br>
│ Analysis utilities for Llama-4 results<br>
│<br>
├── notebooks/<br>
│ ├── exploratory_analysis.ipynb<br>
│ │ Interactive exploration and visualization of results<br>
│ └── bias_summary.ipynb<br>
│ Aggregated summaries for reporting<br>
│<br>
└── utils/<br>
└── helpers.py<br>

---

*(Exact file names may vary slightly as the project evolves.)*

---

## Outputs

The primary outputs used for analysis and reporting are:

- `claude3results-sanitized.json`
- `llama4results-sanitized.json`

Each record contains:
- Prompt metadata (age, gender, role, domain)
- Initial model response
- Justification (“Why?”) response

---

## Intended Use

This repository is designed to support:
- Human coding of gender bias in LLM outputs
- Comparative analysis between models
- Reproducible evaluation of recommendation behavior under controlled conditions

---

## License / Course Context

This project was developed as part of **ECE 59500EAi: AI Ethics & Society** at Purdue University and is intended for academic research and educational use.
