# DeepTrans

An experimental neural code translator developed by SKLP, ICT, CAS

## What's about

a neural code translator supporting various remote LLMs, local LLMs and finetuned LLMs, with careful task decompositions to support real-world repository-level code translation between multiple programming languages.

## Key Features

- Repository-level auto decomposition
- Control-block-level auto function decomposition and translation
- Control-block-level auto unittest generation

## Repository Structure

- `coder`: the core code translation module
- `data`: the dataset module
- `models`: LLM integration module, supporting various LLMs
- `util`: utility module, like logger, config, etc.
- `tools`: python scripts calling external tools
- `tests`: test files for DeepTrans
- `dev`: development files, will be tracked, removed when merged into skeleton
- `tmp`: temporary files, not tracked

## Roadmap

TBD
