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
- kl3m-003
- alea
- legal
- financial
date: '2024-11-07T00:00:00.000Z'
---

# kl3m-004-128k-uncased tokenizer

**NOTE**: This is the same vocabulary as kl3m-004-128k-uncased, but packaged within a `RobertaProcessing` `post_processor` class 
to provide special token handling without loading a custom tokenizer class.

The `kl3m-004-128k-uncased` **case-insensitive** tokenizer is a domain-specific tokenizer trained on a stratified sample of nearly 4M 
documents across general, legal, and financial domains from the `kl3m-data` project, including American English,
British English, Spanish, German, French, Italian, and other common EU languages.  

This tokenizer is being used for the next generation of KL3M embedding and generative models.

Please see `kl3m-001-32k` and `kl3m-003-64k` for the first iteration of our research on domain-specific tokenization.

Note that we are providing both uncased and cased versions of the 128K tokenizer, unlike prior tokenizers, as this was
required to achieve SotA in-domain performance for embedding models on legal and financial text.

## Model Details


### Summary

- **Vocabulary**: 131,072
- **Tokenizer type:** BPE
- **Special token support:** Both causal and masked language modeling
- **Language(s) (NLP):** Primarily English, Spanish, German, French, with a small percentage of other EU languages.
- **Data Sources**: See [`kl3m-data`](https://github.com/alea-institute/kl3m-data) repository.
- **Developed by:** [ALEA Institute](https://aleainstitute.ai).
- **License:** [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)


### Model Description

The `kl3m-004-128k-uncased` tokenizer is a domain-specific tokenizer trained on ~1.5T tokens of financial and legal text from primarily-English sources.

This tokenizer is notable for a number of reasons:

#### Domain Specific

As part of our research on more efficient SLM training for the legal and financial domain, we
trained a domain-specific tokenizer on a large corpus of financial and legal text. This tokenizer
has not, for example, seen any common general pretrain sources like Wikipedia or Common Crawl.

#### Large Added Token Set

As part of our research on efficient and reliable extraction and generation, we inserted
a large numer of deterministic "whole" tokens into the tokenizer, such as HTML tags
like `<span`, common Markdown elements like `#` and `##`, and legal enumerations like `(a)`.

**Note that the kl3m-004-128k-uncased tokenizer has added a number of additional citation formats that were not 
included in the kl3m-001-32k tokenizer.**  These were primarily sourced from empirical data and
the [Free Law Project's reporters-db](https://raw.githubusercontent.com/freelawproject/reporters-db/main/reporters_db/data/),
which were added to the tokenizer to improve model behavior related to legal citations.

See the `get_custom_tokens` method in `kl3m_embeddings/training/kl3m_004/train_tokenizer.py` for
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

Unlike `kl3m-001-32k`, we *do not* retain the space character as a token.  This was done after adding additional legal
citation tokens to the vocabulary, which reduced the number of issues related to space tokenization in legal text.  This
means that the `kl3m-004-128k-uncased` tokenizer uses substantially fewer tokens than `kl3m-001-32k` for most text.

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

## Uses

This tokenizer is intended to be used for English, Spanish, German, or French language text in professional contexts
such as legal and financial documents.

### Recommendations

In general, the `kl3m-004-128k-uncased` tokenizer is recommended over the original `kl3m-001-32k` tokenizer.

```text
Original text:  The Comptroller of the Currency shall have the same authority with respect to functions transferred to the Comptroller of the Currency under the Enhancing Financial Institution Safety and Soundness Act of 2010 as was vested in the Director of the Office of Thrift Supervision on the transfer date, as defined in section 311 of that Act [12 U.S.C. 5411].

kl3m-003-64
-----------
Size:  67
Tokens:  ['The', ' Comptroller', ' of', ' the', ' Currency', ' shall', ' have', ' the', ' same', ' authority', ' with', ' respect', ' to', ' functions', ' transferred', ' to', ' the', ' Comptroller', ' of', ' the', ' Currency', ' under', ' the', ' Enh', 'ancing', ' Financial', ' Institution', ' Safety', ' and', ' ', 'Sound', 'ness', ' Act', ' of', ' 2010', ' as', ' was', ' vested', ' in', ' the', ' Director', ' of', ' the', ' Office', ' of', ' Thrift', ' Supervision', ' on', ' the', ' transfer', ' date', ',', ' as', ' defined', ' in', ' section', ' 311', ' of', ' that', ' Act', ' [', '12', ' ', 'U.S.C.', ' 54', '11', '].']
IDs:  [671, 13273, 295, 281, 25922, 735, 704, 281, 1913, 2451, 440, 1894, 312, 5860, 7264, 312, 281, 13273, 295, 281, 25922, 621, 281, 18926, 4406, 3195, 24448, 5617, 310, 233, 63589, 2130, 854, 295, 1611, 398, 725, 11978, 300, 281, 2827, 295, 281, 1767, 295, 44029, 37141, 395, 281, 3696, 1548, 24, 398, 3011, 300, 782, 6590, 295, 407, 854, 1327, 524, 233, 63761, 3789, 547, 8578]


kl3m-004-128k-uncased
---------------------
Size:  64
Tokens:  ['the', ' comptroller', ' of', ' the', ' currency', ' shall', ' have', ' the', ' same', ' authority', ' with', ' respect', ' to', ' functions', ' transferred', ' to', ' the', ' comptroller', ' of', ' the', ' currency', ' under', ' the', ' enhancing', ' financial', ' institution', ' safety', ' and', ' soundness', ' act', ' of', ' 2010', ' as', ' was', ' vested', ' in', ' the', ' director', ' of', ' the', ' office', ' of', ' thrift', ' supervision', ' on', ' the', ' transfer', ' date', ',', ' as', ' defined', ' in', ' section', ' 311', ' of', ' that', ' act', ' [', '12', ' ', 'u.s.c.', ' 54', '11', '].']
IDs: [536, 16356, 292, 281, 4272, 460, 628, 281, 1552, 1545, 397, 882, 309, 4378, 4032, 309, 281, 16356, 292, 281, 4272, 539, 281, 21164, 1271, 3843, 2737, 313, 35934, 638, 292, 2371, 363, 611, 5286, 298, 281, 2456, 292, 281, 1652, 292, 25900, 7290, 390, 281, 1397, 643, 24, 363, 1921, 298, 590, 12646, 292, 384, 638, 745, 629, 233, 128952, 3834, 571, 4442]  
```

## How to Get Started with the Model

Use the code below to get started with the model.

```
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained('alea-institute/kl3m-004-128k-uncased')
```

## Citation

Tokenizer and dataset publications are pending.

## Contact

For any questions, please contact [ALEA Institute](https://aleainstitute.ai) at [hello@aleainstitute.ai](mailto:hello@aleainstitute.ai) or
create an issue on this repository or [GitHub](https://github.com/alea-institute/kl3m-embedding-research).

![logo](https://aleainstitute.ai/images/alea-logo-ascii-1x1.png)
