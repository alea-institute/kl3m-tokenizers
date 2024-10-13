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


## Roadmap

* [x] **kl3m-001-32k**: [README](kl3m-001-32k/README.md)  |  [Hugging Face](https://huggingface.co/alea-institute/kl3m-001-32k) | Original KL3M tokenizer (Nov 2023)
* [x] **kl3m-001-64k**: [README](kl3m-003-64k/README.md)  |  [Hugging Face](https://huggingface.co/alea-institute/kl3m-003-64k) | Updated KL3M tokenizer (March 2024)
* [ ] **kl3m-004-128k**: In progress

## Examples

You can generate your own comparisons against `tiktoken` like this:
```bash
poetry run python3 examples/compare_tokenizers.py
```

### Statute
```
+=============+============+================+
| gpt-4o      | kl3m-001   | kl3m-003       |
+=============+============+================+
| The         | The        | The            |
+-------------+------------+----------------+
|  Compt      |            | ĠComptroller   |
+-------------+------------+----------------+
| roller      | Comp       | Ġof            |
+-------------+------------+----------------+
|  of         | troller    | Ġthe           |
+-------------+------------+----------------+
|  the        |            | ĠCurrency      |
+-------------+------------+----------------+
|  Currency   | of         | Ġshall         |
+-------------+------------+----------------+
|  shall      |            |                |
+-------------+------------+----------------+
|             | the        |                |
+-------------+------------+----------------+
|             |            |                |
+-------------+------------+----------------+
|             | C          |                |
+-------------+------------+----------------+
|             | urrency    |                |
+-------------+------------+----------------+
|             |            |                |
+-------------+------------+----------------+
|             | shall      |                |
+=============+============+================+
```

### Contract

```
+===============+================+================+
| gpt-4o        | kl3m-001-32k   | kl3m-003-64k   |
+===============+================+================+
| This          | This           | This           |
+---------------+----------------+----------------+
|  Securities   |                | ĠSecurities    |
+---------------+----------------+----------------+
|  Purchase     | Securities     | ĠPurchase      |
+---------------+----------------+----------------+
|  Agreement    |                | ĠAgreement     |
+---------------+----------------+----------------+
|  (            | Purchase       | Ġ(             |
+---------------+----------------+----------------+
| this          |                | this           |
+---------------+----------------+----------------+
|  "            | Agreement      | Ġ"             |
+---------------+----------------+----------------+
| Agreement     |                | Agreement      |
+---------------+----------------+----------------+
| ")            | (              | ")             |
+---------------+----------------+----------------+
|  is           | this           | Ġis            |
+---------------+----------------+----------------+
|  dated        |                | Ġdated         |
+---------------+----------------+----------------+
|  as           | "              | Ġas            |
+---------------+----------------+----------------+
|  of           | Agreement      | Ġof            |
+---------------+----------------+----------------+
|  November     | ")             | ĠNovember      |
+---------------+----------------+----------------+
|               |                | Ġ21            |
+---------------+----------------+----------------+
| 21            | is             | ,              |
+---------------+----------------+----------------+
| ,             |                | Ġ2017          |
+---------------+----------------+----------------+
|               | dated          |                |
+---------------+----------------+----------------+
| 201           |                |                |
+---------------+----------------+----------------+
| 7             | as             |                |
+---------------+----------------+----------------+
|               |                |                |
+---------------+----------------+----------------+
|               | of             |                |
+---------------+----------------+----------------+
|               |                |                |
+---------------+----------------+----------------+
|               | November       |                |
+---------------+----------------+----------------+
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
## License

This ALEA project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions about using this ALEA project, please [open an issue](https://github.com/alea-institute/kl3m-tokenizers/issues) on GitHub.

## Learn More

To learn more about ALEA and its software and research projects like KL3M, visit the [ALEA website](https://aleainstitute.ai/).
