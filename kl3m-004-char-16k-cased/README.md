---
language:
- en
- es
- fr
- de
library_name: tokenizers
license: cc-by-4.0
tags:
- kl3m
- kl3m-004
- alea
- legal
- financial
date: '2024-12-30T00:00:00.000Z'
---

# kl3m-004-char-16k-cased

The `kl3m-004-char-16k-cased` **case-sensitive** tokenizer is a domain-specific **character-based** tokenizer trained 
on a stratified sample of nearly 2M documents across general, legal, and financial domains from the `kl3m-data` project, 
including American English, British English, Spanish, German, French, Italian, and other common EU languages.

This tokenizer uses the standard Byte-Pair Encoding (BPE) tokenizer from `tokenizers`/`transformers`, but modifies the 
training process to restrict the vocabulary to tokens that are at most 3 characters long. Models trained with this tokenizer
should be able to handle a number of use cases that are otherwise difficult to handle with standard tokenizers, such as
low-resource spell-checking, OCR correction, whitespace normalization, and other tasks that require a high degree of character-level
granularity.

## Model Details

### Summary

- **Vocabulary**: 16,384 tokens
- **Tokenizer type:** BPE with 1-4 character tokens
- **Special token support:** Both causal and masked language modeling
- **Language(s) (NLP):** Primarily English, Spanish, German, French, with a small percentage of other EU languages.
- **Data Sources**: See [`kl3m-data`](https://github.com/alea-institute/kl3m-data) repository.
- **Developed by:** [ALEA Institute](https://aleainstitute.ai).
- **License:** [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)

For more information about the `kl3m-004` tokenizers, see the [kl3m-004-128k-cased tokenizer](https://huggingface.co/alea-institute/kl3m-004-128k-cased).

#### Special Tokens for both Embedding and Generative Models

For both training and inference efficiency, we intended this tokenizer vocabulary to be
usable for both embedding and generative models. As such, we included special tokens
suitable for both causal and masked language modeling tasks.

* `<|start|>`: `0`
* `<|end|>`: `1`
* `<|pad|>`: `2`
* `<|unk|>`: `3`
* `<|sep|>`: `4`
* `<|cls|>`: `5`
* `<|mask|>`: `6`

We also added a number of chat and instruction tokens that were not included in `kl3m-001-32k`, including:

* `<|system|>`: `7`
* `</|system|>`: `8`
* `<|user|>`: `9`
* `</|user|>`: `10`
* `<|instruction|>`: `11`
* `</|instruction|>`: `12`

These tokens are identical to those used in the `kl3m-003-64k` tokenizer. 

### Replication

The entire data collection and preprocesing pipeline is being made available, along with
training data, as part of the [ALEA Institute](https://aleainstitute.ai) [KL3M project](https://aleainstitute.ai/work/kl3m/).

The source code to used to train the tokenizer is available on GitHub at:
[https://github.com/alea-institute/kl3m-embedding-research](https://github.com/alea-institute/kl3m-embedding-research)

The data pipeline will be available on GitHub and S3 in the near future.

This specific tokenizer was trained using the following command:

```bash
PYTHONPATH=. poetry run python3 \
  kl3m_tokenizers/tokenizers/kl3m_004/train_char_tokenizer.py \
  --min_frequency 1000 \
  --vocab_size 16384 \
  --pad2 \
  --max_chars 4 \
  sample.20241223173012.jsonl.gz \
  ./kl3m-004-char-16k-cased/
```

```text
Training tokenizer.
[00:33:12] Pre-processing sequences       █████████████████████████████████████████████████████████████ 1849344  /        0
[00:33:32] Pre-processing sequences       █████████████████████████████████████████████████████████████ 0        /        0
[00:00:21] Tokenize words                 █████████████████████████████████████████████████████████████ 20286360 / 20286360
[00:01:01] Count pairs                    █████████████████████████████████████████████████████████████ 20286360 / 20286360
[00:12:39] Compute merges                 █████████████████████████████████████████████████████████████ 16036    /    16036
Adding power-of-2 padding tokens.
Padded vocab to 16384 tokens.
Special tokens: 13
Power-of-2 pad tokens: 13
Final vocab size: 16384
Training time: 2863.67 seconds
Output path: kl3m-004-char-16k-cased
```

### Uses
This tokenizer is intended to be used for English, Spanish, German, or French language tasks where 
character-level details are important, such as OCR correction, spell-checking, or tasks where word boundaries
are not well-defined.  

For a standard BPE "word" tokenizer with a larger vocabulary size, consider using the `kl3m-004-128k-cased` or 
`kl3m-004-128k-uncased` tokenizers.

### Recommendations
The kl3m-004-char-16k-cased tokenizer may be particularly useful when character-level details are important but
resource constraints are not as severe.  For smaller vocabularies with better resource efficiency, consider using the
kl3m-004-char-4k-cased or kl3m-004-char-8k-cased tokenizers.

### How to Get Started with the Model
Use the code below to get started with the model.

```
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained('alea-institute/kl3m-004-char-16k-cased')
```

### Citation
Tokenizer and dataset publications are pending.

## Contact

For any questions, please contact [ALEA Institute](https://aleainstitute.ai) at [hello@aleainstitute.ai](mailto:hello@aleainstitute.ai) or
create an issue on this repository or [GitHub](https://github.com/alea-institute/kl3m-embedding-research).

![logo](https://aleainstitute.ai/images/alea-logo-ascii-1x1.png)
