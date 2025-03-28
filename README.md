# KL3M Tokenizers

## Description
This ALEA project contains the research pipeline and output artifacts for the
KL3M tokenizers, which are used as part of the KL3M family of embedding and generative models.

Read more below to learn more about how the KL3M tokenizers were trained and how they are different.

## Use

If you just want to **use** these tokenizers, you can find them available from the Hugging Face Hub
like most other tokenizers:

**python** with `transformers`:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("alea-institute/kl3m-003-64k")
```

**python** with `tokenizers`:
```python
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_pretrained("alea-institute/kl3m-003-64k")
```

## Technical Details

Like many other tokenizers, the KL3M tokenizers are BPE tokenizers trained with
the `tokenizers` library.  However, unlike most other tokenizers:

1. The KL3M tokenizers were trained on data sources that are free of copyright or licensing issues.
2. The KL3M tokenizers were trained primarily on legal, financial, and governmental works, resulting in:
    * A vocabulary that is better aligned with professional use cases.
    * A vocabulary that is less likely to include toxic or informal language.
3. The KL3M tokenizers include a large number of controlled tokens related to parsing common formats like JSON or HTML.
4. The KL3M tokenizers include a large number of controlled tokens related to legal citations and financial abbreviations.
5. The KL3M tokenizers include support for a variety of tasks including causal (generative) and masked (embedding) tasks.

We also build specialized tokenizers for different use cases:

1. **Character tokenizers**: Designed for low-level tasks like spelling correction or OCR correction.
2. **Pretokenized tokenizers (KL3M-005)**: Using advanced pretokenizers for multi-word tokens:
   * **RandomWhitespaceSplit**: Randomly splits text at whitespace with a configurable probability, allowing the model to learn word combinations.
   * **RandomChunkSplit**: Randomly splits text into chunks of configurable length, creating tokens that cross word boundaries.

These pretokenizers help address common tokenization issues like the "token healing" problem and allow more efficient encoding of common multi-word expressions.


## Roadmap

* [x] **kl3m-001-32k**: [README](kl3m-001-32k/README.md)  |  [Hugging Face](https://huggingface.co/alea-institute/kl3m-001-32k) | Original KL3M tokenizer (Nov 2023)
* [x] **kl3m-003-64k**: [README](kl3m-003-64k/README.md)  |  [Hugging Face](https://huggingface.co/alea-institute/kl3m-003-64k) | Updated KL3M tokenizer (March 2024)
* [x] **kl3m-004-128k-uncased**: [README](kl3m-004-128k-uncased/README.md)  |  [Hugging Face](https://huggingface.co/alea-institute/kl3m-004-128k-uncased) | Updated KL3M tokenizer (November 2024)
* [x] **kl3m-004-128k-cased**: [README](kl3m-004-128k-cased/README.md)  |  [Hugging Face](https://huggingface.co/alea-institute/kl3m-004-128k-cased) | Updated KL3M tokenizer (November 2024)
* [x] **kl3m-004-char-4k-cased**: [README](kl3m-004-char-4k-cased/README.md)  |  [Hugging Face](https://huggingface.co/alea-institute/kl3m-004-char-4k-cased) | Updated KL3M tokenizer (December 2024) 
* [x] **kl3m-004-char-8k-cased**: [README](kl3m-004-char-8k-cased/README.md)  |  [Hugging Face](https://huggingface.co/alea-institute/kl3m-004-char-8k-cased) | Updated KL3M tokenizer (December 2024)
* [x] **kl3m-004-char-16k-cased**: [README](kl3m-004-char-16k-cased/README.md)  |  [Hugging Face](https://huggingface.co/alea-institute/kl3m-004-char-16k-cased) | Updated KL3M tokenizer (December 2024)
* [ ] **kl3m-005-pretokenized**: Tokenizers with RandomWhitespaceSplit and RandomChunkSplit pretokenizers (March 2025)
 
## Examples

### "Normal" Tokenizers

You can generate your own comparisons against `tiktoken` like this:
```bash
poetry run python3 examples/compare_tokenizers.py
```

### Statute
```
+=============+================+================+=======================+=========================+
| gpt-4o      | kl3m-001-32k   | kl3m-003-64k   | kl3m-004-128k-cased   | kl3m-004-128k-uncased   |
+=============+================+================+=======================+=========================+
| The         | The            | The            | The                   | the                     |
+-------------+----------------+----------------+-----------------------+-------------------------+
|  Compt      |                | ĠComptroller   | ĠComptroller          | Ġcomptroller            |
+-------------+----------------+----------------+-----------------------+-------------------------+
| roller      | Comp           | Ġof            | Ġof                   | Ġof                     |
+-------------+----------------+----------------+-----------------------+-------------------------+
|  of         | troller        | Ġthe           | Ġthe                  | Ġthe                    |
+-------------+----------------+----------------+-----------------------+-------------------------+
|  the        |                | ĠCurrency      | ĠCurrency             | Ġcurrency               |
+-------------+----------------+----------------+-----------------------+-------------------------+
|  Currency   | of             | Ġshall         | Ġshall                | Ġshall                  |
+-------------+----------------+----------------+-----------------------+-------------------------+
|  shall      |                |                |                       |                         |
+-------------+----------------+----------------+-----------------------+-------------------------+
|             | the            |                |                       |                         |
+-------------+----------------+----------------+-----------------------+-------------------------+
|             |                |                |                       |                         |
+-------------+----------------+----------------+-----------------------+-------------------------+
|             | C              |                |                       |                         |
+-------------+----------------+----------------+-----------------------+-------------------------+
|             | urrency        |                |                       |                         |
+-------------+----------------+----------------+-----------------------+-------------------------+
|             |                |                |                       |                         |
+-------------+----------------+----------------+-----------------------+-------------------------+
|             | shall          |                |                       |                         |
+=============+================+================+=======================+=========================+

```

### Contract

```
+===============+================+================+=======================+=========================+
| gpt-4o        | kl3m-001-32k   | kl3m-003-64k   | kl3m-004-128k-cased   | kl3m-004-128k-uncased   |
+===============+================+================+=======================+=========================+
| This          | This           | This           | This                  | this                    |
+---------------+----------------+----------------+-----------------------+-------------------------+
|  Securities   |                | ĠSecurities    | ĠSecurities           | Ġsecurities             |
+---------------+----------------+----------------+-----------------------+-------------------------+
|  Purchase     | Securities     | ĠPurchase      | ĠPurchase             | Ġpurchase               |
+---------------+----------------+----------------+-----------------------+-------------------------+
|  Agreement    |                | ĠAgreement     | ĠAgreement            | Ġagreement              |
+---------------+----------------+----------------+-----------------------+-------------------------+
|  (            | Purchase       | Ġ(             | Ġ(                    | Ġ(                      |
+---------------+----------------+----------------+-----------------------+-------------------------+
| this          |                | this           | this                  | this                    |
+---------------+----------------+----------------+-----------------------+-------------------------+
|  "            | Agreement      | Ġ"             | Ġ"                    | Ġ"                      |
+---------------+----------------+----------------+-----------------------+-------------------------+
| Agreement     |                | Agreement      | Agreement             | agreement               |
+---------------+----------------+----------------+-----------------------+-------------------------+
| ")            | (              | ")             | ")                    | ")                      |
+---------------+----------------+----------------+-----------------------+-------------------------+
|  is           | this           | Ġis            | Ġis                   | Ġis                     |
+---------------+----------------+----------------+-----------------------+-------------------------+
|  dated        |                | Ġdated         | Ġdated                | Ġdated                  |
+---------------+----------------+----------------+-----------------------+-------------------------+
|  as           | "              | Ġas            | Ġas                   | Ġas                     |
+---------------+----------------+----------------+-----------------------+-------------------------+
|  of           | Agreement      | Ġof            | Ġof                   | Ġof                     |
+---------------+----------------+----------------+-----------------------+-------------------------+
|  November     | ")             | ĠNovember      | ĠNovember             | Ġnovember               |
+---------------+----------------+----------------+-----------------------+-------------------------+
|               |                | Ġ21            | Ġ21                   | Ġ21                     |
+---------------+----------------+----------------+-----------------------+-------------------------+
| 21            | is             | ,              | ,                     | ,                       |
+---------------+----------------+----------------+-----------------------+-------------------------+
| ,             |                | Ġ2017          | Ġ2017                 | Ġ2017                   |
+---------------+----------------+----------------+-----------------------+-------------------------+
|               | dated          |                |                       |                         |
+---------------+----------------+----------------+-----------------------+-------------------------+
| 201           |                |                |                       |                         |
+---------------+----------------+----------------+-----------------------+-------------------------+
| 7             | as             |                |                       |                         |
+---------------+----------------+----------------+-----------------------+-------------------------+
|               |                |                |                       |                         |
+---------------+----------------+----------------+-----------------------+-------------------------+
|               | of             |                |                       |                         |
+---------------+----------------+----------------+-----------------------+-------------------------+
|               |                |                |                       |                         |
+---------------+----------------+----------------+-----------------------+-------------------------+
|               | November       |                |                       |                         |
+---------------+----------------+----------------+-----------------------+-------------------------+
|               |                |                |                       |                         |
+---------------+----------------+----------------+-----------------------+-------------------------+
|               | 21             |                |                       |                         |
+---------------+----------------+----------------+-----------------------+-------------------------+
|               | ,              |                |                       |                         |
+---------------+----------------+----------------+-----------------------+-------------------------+
|               |                |                |                       |                         |
+---------------+----------------+----------------+-----------------------+-------------------------+
|               | 2017           |                |                       |                         |
+===============+================+================+=======================+=========================+
```

### Character Tokenizers

```text
+=======================+=======================+========================+
|   kl3m-004-4k-cased   |   kl3m-004-8k-cased   |   kl3m-004-16k-cased   |
+=======================+=======================+========================+
| KE                    | KE                    | KE                     |
+-----------------------+-----------------------+------------------------+
| GUL                   | GUL                   | G                      |
+-----------------------+-----------------------+------------------------+
| AT                    | AT                    | UL                     |
+-----------------------+-----------------------+------------------------+
| ED                    | ED                    | ATED                   |
+-----------------------+-----------------------+------------------------+
| ĠN                    | ĠN                    | ĠNAT                   |
+-----------------------+-----------------------+------------------------+
| AT                    | AT                    | URAL                   |
+-----------------------+-----------------------+------------------------+
| UR                    | UR                    | ĠGAS                   |
+-----------------------+-----------------------+------------------------+
| AL                    | AL                    | .âĢĶ                   |
+-----------------------+-----------------------+------------------------+
| ĠG                    | ĠG                    | The                    |
+-----------------------+-----------------------+------------------------+
| AS                    | AS                    | Ġt                     |
+-----------------------+-----------------------+------------------------+
| .                     | .                     | Ġe                     |
+-----------------------+-----------------------+------------------------+
| âĢĶ                   | âĢĶ                   | Ġr                     |
+-----------------------+-----------------------+------------------------+
| The                   | The                   | Ġm                     |
+-----------------------+-----------------------+------------------------+
| Ġt                    | Ġt                    | Ġ'                     |
+-----------------------+-----------------------+------------------------+
| Ġe                    | Ġe                    | Ġr                     |
+-----------------------+-----------------------+------------------------+
| Ġr                    | Ġr                    | Ġe                     |
+-----------------------+-----------------------+------------------------+
| Ġm                    | Ġm                    | Ġg                     |
+-----------------------+-----------------------+------------------------+
| Ġ'                    | Ġ'                    | Ġu                     |
+-----------------------+-----------------------+------------------------+
| Ġr                    | Ġr                    | Ġl                     |
+-----------------------+-----------------------+------------------------+
| Ġe                    | Ġe                    | Ġa                     |
+-----------------------+-----------------------+------------------------+
| Ġg                    | Ġg                    | Ġt                     |
+-----------------------+-----------------------+------------------------+
| Ġu                    | Ġu                    | Ġe                     |
+-----------------------+-----------------------+------------------------+
| Ġl                    | Ġl                    | Ġd                     |
+-----------------------+-----------------------+------------------------+
| Ġa                    | Ġa                    | Ġn                     |
+-----------------------+-----------------------+------------------------+
| Ġt                    | Ġt                    | Ġa                     |
+-----------------------+-----------------------+------------------------+
| Ġe                    | Ġe                    | Ġt                     |
+-----------------------+-----------------------+------------------------+
| Ġd                    | Ġd                    | Ġu                     |
+-----------------------+-----------------------+------------------------+
| Ġn                    | Ġn                    |                        |
+-----------------------+-----------------------+------------------------+
| Ġa                    | Ġa                    |                        |
+-----------------------+-----------------------+------------------------+
| Ġt                    | Ġt                    |                        |
+-----------------------+-----------------------+------------------------+
| Ġu                    | Ġu                    |                        |
+=======================+=======================+========================+
```

## Training a New Tokenizer

You can replicate the training process for tokenizers like this:

**kl3m-001-32k**:
```bash
$ PYTHONPATH=. poetry run python3 kl3m_tokenizers/tokenizers/kl3m_001/train_tokenizer.py --vocab_size 4096 --pad2 samples/usc.1000.jsonl.gz tokenizer-4k
```

**Output**
```
[00:00:00] Pre-processing sequences       ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 0        /        0[00:00:00] Tokenize words                 ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 31728    /    31728
[00:00:00] Count pairs                    ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 31728    /    31728
[00:00:00] Compute merges                 ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 2204     /     2204
Added 1612 custom tokens to size=4023.
Adding power-of-2 padding tokens.
Padded vocab to 4096 tokens.
Special tokens: 7
Custom tokens: 1612
Power-of-2 pad tokens: 73
Final vocab size: 4096
Training time: 0.67 seconds
Output path: tokenizer-4k/
```

**kl3m-003-64k**:
```bash
$ PYTHONPATH=. poetry run python3 kl3m_tokenizers/tokenizers/kl3m_003/train_tokenizer.py --vocab_size 8192 --pad2 samples/usc.1000.jsonl.gz tokenizer-8k
```

**Output**
```
Training tokenizer.
[00:00:00] Pre-processing sequences       ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 9788     /        0Finished parsing samples/usc.1000.jsonl.gz: 10000 records
[00:00:00] Pre-processing sequences       ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 0        /        0[00:00:00] Tokenize words                 ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 31699    /    31699
[00:00:00] Count pairs                    ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 31699    /    31699
[00:00:00] Compute merges                 ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 3472     /     3472
Adding custom tokens.
Adding power-of-2 padding tokens.
Padded vocab to 8192 tokens.
Special tokens: 13
Custom tokens: 4393
Power-of-2 pad tokens: 58
Final vocab size: 8192
Training time: 1.31 seconds
Output path: tokenizer-8k/
```

**kl3m-005 with RandomWhitespaceSplit Pretokenizer**:
```bash
$ PYTHONPATH=. poetry run python3 kl3m_tokenizers/tokenizers/kl3m_005/train_tokenizer.py --vocab_size 8192 --pad2 --pretokenizer whitespace --split_probability 0.3 samples/usc.1000.jsonl.gz tokenizer-whitespace-8k
```

**kl3m-005 with RandomChunkSplit Pretokenizer**:
```bash
$ PYTHONPATH=. poetry run python3 kl3m_tokenizers/tokenizers/kl3m_005/train_tokenizer.py --vocab_size 8192 --pad2 --pretokenizer chunk --min_length 2 --max_length 5 samples/usc.1000.jsonl.gz tokenizer-chunk-8k
```
## License

This ALEA project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions about using this ALEA project, please [open an issue](https://github.com/alea-institute/kl3m-tokenizers/issues) on GitHub.

## Learn More

To learn more about ALEA and its software and research projects like KL3M, visit the [ALEA website](https://aleainstitute.ai/).
