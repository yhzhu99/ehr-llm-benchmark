# EHR-LLM-Benchmark

This repository contains the code and resources for our paper "Is larger always better? Evaluating and prompting large language models for non-generative medical tasks".

Our study provides a comprehensive benchmark of various models, including GPT-based LLMs, BERT-based language models, and conventional clinical predictive models to determine their efficacy in non-generative medical tasks.

## 🎯 Prediction Tasks

The following prediction tasks have been implemented in this repository:

### [Structured EHR Data Tasks](structured_ehr/README.md)

- In-hospital Mortality Prediction
- 30-day Readmission Prediction

### [Unstructured clinical notes tasks](unstructured_note/README.md)

- Medical Sentence Matching
- ICD Code Clustering

## 🚀 Model Zoo

We examine a diverse array of models including GPT-based LLMs, BERT-based models, and conventional clinical predictive models:

### GPT-based LLMs

- GPT-2
- GPT-3.5
- GPT-4
- LLama-3
- BioGPT
- MedAlpaca
- HuatuoGPT-II
- Meditron
- OpenBioLLM.

### BERT-based Models

- BERT
- BioBERT
- ClinicalBERT
- GatorTron
- Clinical-Longformer

### Conventional Clinical Predictive Models

- Decision tree (DT)
- XGBoost
- GRU
- ConCare
- GRASP
- M3Care
- AICare

## 🗄️ Repository Structure

- `structured_ehr/`: Codes for structured EHR tasks.
- `unstructured_note/`: Codes for unstructured clinical notes tasks.

## ⚙️ Requirements

To get started with the repository, ensure your environment meets the following requirements. We use `uv` as our Python package manager with Python 3.12.

- Run `uv sync`
