# DeepTrans

An experimental neural code translator developed by SKLP, ICT, CAS

## What's about

a neural code translator supporting various remote LLMs, local LLMs and finetuned LLMs, with careful task decompositions to support real-world repository-level code translation between multiple programming languages.

**Must read**: Difference with [TRANSAGENT: An LLM-Based Multi-Agent System for Code Translation](http://arxiv.org/abs/2409.19894), a October arxiv paper by FDU:

1. they use agent as their solution, described in a software-engineering way, in 4 steps: raw translation, fix syntax error, alignment with block, fix bug located in blocks
2. they only split the program when do the bug fix period(step3 and step4).
3. they consider the value dump way similar to us.
4. they evaluate the results on 600 functions across C++/Python/Java.
5. TBD.

## Key Features

- Repository-level auto decomposition
- Control-block-level auto function decomposition and translation
- Control-block-level auto unittest generation
- Auto debugging with exit value in control-blocks

**potential plans** to enhance the work:

- Multimodal specifications, by a paper [SPECTRA: Enhancing the Code Translation Ability of Language Models by Generating Multi-Modal Specifications](http://arxiv.org/abs/2405.18574), maybe graph input can be integrated as well.
- Constrained generation, can follow similar practices from paper136.
- Input-centric characterization, what if the inputs are very large? like paper170.
- Specific focus on language properties, like OOP features, or type system(rust for example, see paper511).

## Repository Structure

- `coder`: the core code translation module
- `data`: the dataset module
- `models`: LLM integration module, supporting various LLMs
- `util`: utility module, like logger, config, etc.
- `tools`: python scripts calling external tools
- `tests`: test files for DeepTrans
- `dev`: development files, will be tracked, removed when merged into skeleton
- `tmp`: temporary files, not tracked

## Evaluation

### Transcoder

1. Transcoder is a **cpp-java-python** parallel dataset in function-level, without explicit unittest.
2. therefore, as a code translator with **soundness** guarantee, we need to first synthesize valid unittest for all transcoder cases: to do so, we just need to run the unittest generator for each code(ground truth) individually, and collect them.
3. After unittest generation, we will translate code from one language into another language, and test the behavioral correctness (pass@1, pass@k, pass@k@n) using our synthesized unittests.

### UniTrans

TBD

### TransAgent

TBD

## Roadmap

TBD
