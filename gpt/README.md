# Advanced nano-gpt

## Prepare text data

### Shakespeare

If you choose character-based tokenization, run `data/shakespeare_char/prepare.ipynb`.

If you choose gpt2-like bpe tokenization, run `data/shakespeare_bpe/prepare.ipynb`.

### OpenWebTextCorpus

If you want to use the original 9M `Skylion007/openwebtext`, you need to be careful about your package versions:

```Bash
pip install "datasets<4.0.0"
pip install "huggingface_hub<0.25"
```
