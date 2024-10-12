---
library_name: tokenizers
tags:
- kl3m
- kl3m-001
- alea
- legal
- financial
date: '2023-12-28T00:00:00.000Z'
license: cc-by-4.0
language:
- en
---

# kl3m-001-32k tokenizer

The `kl3m-001-32k` tokenizer is a domain-specific tokenizer trained on ~500B tokens of financial and legal text from primarily-English sources.

This tokenizer was used for the first generation of KL3M embedding and generative models, including
`kl3m-170M`, `kl3m-1.7B`, `kl3m-embedding-001`, and `kl3m-embedding-002`.

Please see `kl3m-003-64k` for the next iteration of our research on domain-specific tokenization.

## Model Details


### Summary

- **Vocabulary**: 32,768
- **Tokenizer type:** BPE
- **Special token support:** Both causal and masked language modeling
- **Language(s) (NLP):** English
- **Developed by:** Originally by [273 Ventures LLC](https://273ventures.com), donated to [ALEA Institute](https://aleainstitute.ai).
- **License:** [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)


### Model Description

The `kl3m-001-32k` tokenizer is a domain-specific tokenizer trained on ~500B tokens of financial and legal text from primarily-English sources.

This tokenizer is notable for a number of reasons:

#### Domain Specific

As part of our research on more efficient SLM training for the legal and financial domain, we
trained a domain-specific tokenizer on a large corpus of financial and legal text. This tokenizer
has not, for example, seen any common general pretrain sources like Wikipedia or Common Crawl.

#### Large Added Token Set

As part of our research on efficient and reliable extraction and generation, we inserted
a large numer of deterministic "whole" tokens into the tokenizer, such as HTML tags
like `<span`, common Markdown elements like `#` and `##`, and legal enumerations like `(a)`.

See the `get_custom_tokens` method in `kl3m_embeddings/training/kl3m_001/train_tokenizer.py` for
more details:

```python
def get_custom_tokens(
    include_whitespace: bool = True,
    include_markdown: bool = True,
    include_html: bool = True,
    include_json: bool = True,
    include_xml: bool = True,
    include_years: bool = True,
    include_citations: bool = True,
    lowercase: bool = False,
) -> list[str]:
```

#### Space Preservation

Unlike many tokenizers, we retain the space character as a token after early small-scale experiments.
While this has substantial space implications for some types of text with many shorter words, we found
that it reduced the rate of a number of undesirable phenomena.

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

### Replication

The entire data collection and preprocesing pipeline is being made available, along with
training data, as part of the [ALEA Institute](https://aleainstitute.ai) [KL3M project](https://aleainstitute.ai/work/kl3m/).

The source code to used to train the tokenizer is available on GitHub at:
[https://github.com/alea-institute/kl3m-embedding-research](https://github.com/alea-institute/kl3m-embedding-research)

The data pipeline will be available on GitHub and S3 in the near future.

## Uses

This tokenizer is intended to be used for English language text in professional contexts such as legal and financial documents.

### Recommendations

Please see the `kl3m-003-64k` tokenizer for the next iteration of our research on domain-specific tokenization.

In general, the `kl3m-003-64k` tokenizer is recommended over the original `kl3m-001-32k` tokenizer.

```text
Original text:  The Comptroller of the Currency shall have the same authority with respect to functions transferred to
 the Comptroller of the Currency under the Enhancing Financial Institution Safety and Soundness Act of 2010 as was
 vested in the Director of the Office of Thrift Supervision on the transfer date, as defined in section 311 of that
 Act [12 U.S.C. 5411].

kl3m-001-32k
--------------------
Size:  147
Tokens:  ['The', ' ', 'Comp', 'troller', ' ', 'of', ' ', 'the', ' ', 'C', 'urrency', ' ', 'shall', ' ', 'have', ' ', 'the', ' ', 'same', ' ', 'authority', ' ', 'with', ' ', 'respect', ' ', 'to', ' ', 'fun', 'ctions', ' ', 'transferred', ' ', 'to', '\n', ' ', 'the', ' ', 'Comp', 'troller', ' ', 'of', ' ', 'the', ' ', 'C', 'urrency', ' ', 'under', ' ', 'the', ' ', 'En', 'ha', 'ncing', ' ', 'Financial', ' ', 'Institution', ' ', 'Sa', 'fe', 'ty', ' ', 'a', 'n', 'd', ' ', 'S', 'ound', 'ness', ' ', 'Act', ' ', 'of', ' ', '2010', ' ', 'as', ' ', 'was', '\n', ' ', 'vested', ' ', 'i', 'n', ' ', 'the', ' ', 'Director', ' ', 'of', ' ', 'the', ' ', 'Office', ' ', 'of', ' ', 'Th', 'rift', ' ', 'Superv', 'ision', ' ', 'o', 'n', ' ', 'the', ' ', 'transfer', ' ', 'date', ',', ' ', 'as', ' ', 'defined', ' ', 'i', 'n', ' ', 'section', ' ', '311', ' ', 'of', ' ', 'that', '\n', ' ', 'Act', ' ', '[', '12', ' ', 'U', '.', 'S', '.', 'C', '.', ' ', '54', '11', '].']
IDs:  [815, 31673, 3546, 14529, 31673, 269, 31673, 441, 31673, 41, 9646, 31673, 5516, 31673, 4130, 31673, 441, 31673, 8685, 31673, 14765, 31673, 1946, 31673, 12500, 31673, 265, 31673, 12122, 1935, 31673, 12677, 31673, 265, 31674, 31673, 441, 31673, 3546, 14529, 31673, 269, 31673, 441, 31673, 41, 9646, 31673, 2823, 31673, 441, 31673, 1871, 288, 2655, 31673, 20796, 31673, 29543, 31673, 4778, 362, 1004, 31673, 71, 84, 74, 31673, 57, 1098, 1647, 31673, 8494, 31673, 269, 31673, 3629, 31673, 310, 31673, 3182, 31674, 31673, 9761, 31673, 79, 84, 31673, 441, 31673, 21209, 31673, 269, 31673, 441, 31673, 8827, 31673, 269, 31673, 788, 11004, 31673, 28799, 873, 31673, 85, 84, 31673, 441, 31673, 12790, 31673, 2726, 18, 31673, 310, 31673, 10212, 31673, 79, 84, 31673, 3517, 31673, 15340, 31673, 269, 31673, 1704, 31674, 31673, 8494, 31673, 65, 534, 31673, 59, 20, 57, 20, 41, 20, 31673, 2195, 572, 5582]

kl3m-003-64k
--------------------
Size:  70
Tokens:  ['The', 'ĠComptroller', 'Ġof', 'Ġthe', 'ĠCurrency', 'Ġshall', 'Ġhave', 'Ġthe', 'Ġsame', 'Ġauthority', 'Ġwith', 'Ġrespect', 'Ġto', 'Ġfunctions', 'Ġtransferred', 'Ġto', 'Ċ', 'Ġthe', 'ĠComptroller', 'Ġof', 'Ġthe', 'ĠCurrency', 'Ġunder', 'Ġthe', 'ĠEnh', 'ancing', 'ĠFinancial', 'ĠInstitution', 'ĠSafety', 'Ġand', 'Ġ', 'Sound', 'ness', 'ĠAct', 'Ġof', 'Ġ2010', 'Ġas', 'Ġwas', 'Ċ', 'Ġvested', 'Ġin', 'Ġthe', 'ĠDirector', 'Ġof', 'Ġthe', 'ĠOffice', 'Ġof', 'ĠThrift', 'ĠSupervision', 'Ġon', 'Ġthe', 'Ġtransfer', 'Ġdate', ',', 'Ġas', 'Ġdefined', 'Ġin', 'Ġsection', 'Ġ311', 'Ġof', 'Ġthat', 'Ċ', 'ĠAct', 'Ġ[', '12', 'Ġ', 'U.S.C.', 'Ġ54', '11', '].']
IDs:  [671, 13273, 295, 281, 25922, 735, 704, 281, 1913, 2451, 440, 1894, 312, 5860, 7264, 312, 211, 281, 13273, 295, 281, 25922, 621, 281, 18926, 4406, 3195, 24448, 5617, 310, 233, 63589, 2130, 854, 295, 1611, 398, 725, 211, 11978, 300, 281, 2827, 295, 281, 1767, 295, 44029, 37141, 395, 281, 3696, 1548, 24, 398, 3011, 300, 782, 6590, 295, 407, 211, 854, 1327, 524, 233, 63761, 3789, 547, 8578]

```

## How to Get Started with the Model

Use the code below to get started with the model.

```
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained('alea-institute/kl3m-001-32k')
```

## Citation

Tokenizer and dataset publications are pending.

## Contact

For any questions, please contact [ALEA Institute](https://aleainstitute.ai) at [hello@aleainstitute.ai](mailto:hello@aleainstitute.ai) or
create an issue on this repository or [GitHub](https://github.com/alea-institute/kl3m-embedding-research).

![logo](https://aleainstitute.ai/images/alea-logo-ascii-1x1.png)
