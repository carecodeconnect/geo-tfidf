# `geo-tfidf`

Quick demo for Generative Engine Optimisation (GEO) mini-project using classical Natural Language Processing (NLP), featuring Term Frequency Inverse Document Frequency (TFIDF).

Inspired by the original [GEO](https://arxiv.org/abs/2311.09735) study by Aggarwal *et al.* (2023).

For this project, we analyse the [AG News](https://huggingface.co/datasets/wangrongsheng/ag_news) dataset.

## Dev

For development work:

```bash
# Add ipykernel as a dev dependency
uv add --dev ipykernel
# Register kernel with ipykernel by installing kernelspec
uv run python -m ipykernel install --user --name geo-tfidf --display-name "Python (geo-tfidf)"
# Test
uv run jupyter kernelspec list
```