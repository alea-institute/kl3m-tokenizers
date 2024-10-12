"""
Train a BPE tokenizer for KL3M-003+ with all custom and standard tokens.
"""

# preserving copies as-is without refactoring for historical replication
# pylint: disable=duplicate-code

# imports
import argparse
import gzip
import json
import random
import string
import time
from math import log2
from pathlib import Path
from typing import Generator

# packages
import httpx
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.normalizers import NFC, NFD, NFKC, NFKD


# pylint: disable=too-many-positional-arguments,too-many-branches,too-many-statements
# set up custom token data
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
    """
    Get the list of the custom tokens to include if
    automatically inferred from the input data.

    Args:
        include_whitespace (bool, optional): Whether to include whitespace. Defaults to True.
        include_markdown (bool, optional): Whether to include markdown. Defaults to True.
        include_html (bool, optional): Whether to include html. Defaults to True.
        include_json (bool, optional): Whether to include json. Defaults to True.
        include_xml (bool, optional): Whether to include xml. Defaults to True.
        include_years (bool, optional): Whether to include years. Defaults to True.
        include_citations (bool, optional): Whether to include citations. Defaults to True.
        lowercase (bool, optional): Whether to lowercase the tokens. Defaults to False.

    Returns:
        list[str]: List of custom tokens
    """
    tokens = []

    if include_whitespace:
        # ad repeated whitespace like 2-8 consecutive space, \t, \r, \n
        tokens.extend(
            [
                whitespace * multiple
                for whitespace in [" ", "\t", "\r", "\n"]
                for multiple in range(2, 9)
            ]
            + [
                "\r\n",
                "\t ",
            ]
        )

    if include_markdown:
        # add markdown tokens
        tokens.extend(
            [
                # markdown heading
                "##",
                "###",
                "####",
                "#####",
                "######",
                # markdown bold
                "**",
                # markdown strikethrough
                "~~",
                # markdown code
                "```",
                # markdown image
                "![",
                # uri components
                "http://",
                "https://",
                "www.",
                "ftp://",
                "mailto:",
                "tel:",
                "://",
            ]
        )

        # add repeated ., -, =, _, *, ~ x2-8
        tokens.extend(
            [
                char * multiple
                for char in [".", "-", "=", "_", "*", "~"]
                for multiple in range(2, 9)
            ]
        )

    if include_html:
        # add opening html tags
        html_tags = [
            "html",
            "head",
            "body",
            "div",
            "span",
            "img",
            "table",
            "thead",
            "tbody",
            "tfoot",
            "tr",
            "td",
            "ul",
            "ol",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "iframe",
            "script",
            "style",
            "svg",
            "input",
            "button",
            "form",
            "label",
            "select",
            "option",
            "textarea",
            "meta",
            "link",
            "title",
            "pre",
            "code",
            "row",
            "column",
        ]

        # create opening tags
        tokens.extend(["<" + tag for tag in html_tags])
        if not lowercase:
            tokens.extend(["<" + tag.upper() for tag in html_tags])

        # create closing tags
        tokens.extend(["</" + tag + ">" for tag in html_tags])

        # create key attributes
        html_attributes = [
            "href",
            "src",
            "alt",
            "title",
            "class",
            "id",
        ]

        # create key attributes
        tokens.extend([" " + attribute + "=" for attribute in html_attributes])
        if not lowercase:
            tokens.extend(
                [" " + attribute.upper() + "=" for attribute in html_attributes]
            )

    if include_json:
        # add mutli-char json tokens
        # create json tokens
        tokens.extend(
            [
                '{"',
                '"}',
                '["',
                '"]',
                '":',
                '","',
                '":"',
                "null",
                "true",
                "false",
            ]
        )

    if include_xml:
        # add mutli-char xml tokens
        # create xml tokens
        tokens.extend(
            [
                "<!--",
                "-->",
                "<![CDATA[",
                "]]>",
                "<?xml",
                "<xml",
                "</xml>",
            ]
        )

    if include_years:
        # add years to reduce hallucination risk
        tokens.extend([str(year) for year in range(1776, 2050)])

    if include_citations:
        # add citation building blocks like (a) for all letters, numbers, and roman numerals
        tokens.extend([f"({letter})" for letter in string.ascii_letters])
        if not lowercase:
            tokens.extend([f"({letter.upper()})" for letter in string.ascii_letters])

        tokens.extend({f"({number})" for number in range(1, 100)})

        tokens.extend(
            [
                f"({roman})"
                for roman in ["ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]
            ]
        )
        if not lowercase:
            tokens.extend(
                [
                    f"({roman.upper()})"
                    for roman in ["ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]
                ]
            )

        # add all numbers up to 999 for section citations
        tokens.extend([str(number) for number in range(10, 1000)])

        # add common abbreviations from judicial, legislative, and regulatory documents
        # sourced from BSD-3 reportersdb
        # https://raw.githubusercontent.com/freelawproject/reporters-db/main/reporters_db/data/laws.json
        laws_url = "https://raw.githubusercontent.com/freelawproject/reporters-db/main/reporters_db/data/laws.json"
        response = httpx.get(laws_url)
        laws_data = json.loads(response.text)
        for key in laws_data.keys():
            tokens.extend(key.split())

        # https://raw.githubusercontent.com/freelawproject/reporters-db/main/reporters_db/data/reporters.json
        reporters_url = "https://raw.githubusercontent.com/freelawproject/reporters-db/main/reporters_db/data/reporters.json"
        response = httpx.get(reporters_url)
        reporters_data = json.loads(response.text)
        for key in reporters_data.keys():
            tokens.extend(key.split())
            try:
                for record in reporters_data[key]:
                    for variation in record["variations"]:
                        tokens.extend(variation.split())
            except Exception:  # pylint: disable=broad-except
                pass

        # https://raw.githubusercontent.com/freelawproject/reporters-db/main/reporters/data/journals.json
        journals_url = "https://raw.githubusercontent.com/freelawproject/reporters-db/main/reporters_db/data/journals.json"
        response = httpx.get(journals_url)
        journals_data = json.loads(response.text)
        for key in journals_data.keys():
            tokens.extend(key.split())

        # https://raw.githubusercontent.com/freelawproject/reporters-db/main/reporters_db/data/case_name_abbreviations.json
        cn_url = "https://raw.githubusercontent.com/freelawproject/reporters-db/main/reporters_db/data/case_name_abbreviations.json"
        response = httpx.get(cn_url)
        cn_data = json.loads(response.text)
        for key in cn_data.keys():
            tokens.extend(key.split())

    # sort and unique
    tokens = sorted(list(set(tokens)))

    # remove any single-length tokens
    tokens = [token for token in tokens if len(token) > 1]

    # return the tokens
    return tokens


# jsonl generator
def yield_field_from_json(
    input_path_list: list[Path],
    field_name: str = "text",
) -> Generator[str, None, None]:
    """
    Yield a field from a JSONL file.

    Args:
        input_path_list (list[Path]): List of paths to the input JSONL files
        field_name (str): Name of the field to yield

    Yields:
        Generator[str, None, None]: Generator of the field
    """
    # yield the field from the JSONL file
    for input_path in input_path_list:
        try:
            # check if there is a :%f in the input path
            if ":" in input_path.name:
                # split it into the file and rate
                path_tokens = str(input_path).split(":", maxsplit=1)
                input_path, rate = Path(path_tokens[0]), path_tokens[1]
            else:
                rate = None

            # get the right opener
            if str(input_path).lower().endswith(".gz"):
                opener = gzip.open
            else:
                opener = open  # type: ignore

            # drop orjsonl for encoding issues
            with opener(input_path, "rb") as jsonl_file:
                line_number = 0
                for line in jsonl_file:
                    line_number += 1
                    if rate:
                        # check if we should yield this record
                        if random.random() > float(rate):
                            continue

                    # parse the record
                    try:
                        record = json.loads(line, strict=False)
                    except Exception as json_error:  # pylint: disable=broad-except
                        print(
                            f"Unable to parse line {line_number} from {input_path}: {json_error}"
                        )
                        continue

                    if field_name in record:
                        try:
                            if isinstance(record[field_name], str):
                                # ensure we don't have any ValueError: Unpaired high surrogate when decoding issues
                                yield (
                                    record[field_name]
                                    .encode("utf-8", errors="replace")
                                    .decode("utf-8", errors="replace")
                                )
                            elif isinstance(record[field_name], bytes):
                                yield record[field_name].decode(
                                    "utf-8", errors="replace"
                                )
                        except Exception as record_error:  # pylint: disable=broad-except
                            print(
                                f"Unable to parse line from {input_path}: {record_error}"
                            )
                            continue

            print(f"Finished parsing {input_path}: {line_number} records")
        except Exception as json_error:  # pylint: disable=broad-except
            print(f"Unable to parse {input_path}: {json_error}")
            continue


def train_tokenizer(
    input_path_list: list[Path],
    output_path: Path,
    vocab_size: int = 32768,
    min_frequency: int = 10,
    normalization: str | None = "nfkc",
    lowercase: bool = False,
    pad2: bool = False,
):
    """
    Train a GPT NeoX tokenizer.

    Args:
        input_path_list (list[Path]): List of paths to the input JSONL files
        output_path (Path): Path to the output tokenizer directory
        vocab_size (int, optional): Size of the vocabulary. Defaults to 32768.
        min_frequency (int, optional): Minimum frequency of a token to be included in the vocabulary. Defaults to 5.
        normalization (str, optional): Normalization to apply to the text. Defaults to None.
        lowercase (bool, optional): Whether to lowercase the text. Defaults to False.
        pad2 (bool, optional): Whether to pad the vocab to the next power of 2. Defaults to False.
    """
    # start time
    start_time = time.time()

    # setup the special token list
    special_tokens = [
        # start
        "<|start|>",
        # end
        "<|end|>",
        # pad
        "<|pad|>",
        # include these for possible compatibility or future use in paired enc-dec
        "<|unk|>",
        # sep
        "<|sep|>",
        # cls
        "<|cls|>",
        # mask
        "<|mask|>",
        # system, user, and instruction start and end tags
        "<|system|>",
        "</|system|>",
        "<|user|>",
        "</|user|>",
        "<|instruction|>",
        "</|instruction|>",
    ]

    # initialize the BPE tokenizer with casing
    tokenizer = ByteLevelBPETokenizer(
        lowercase=lowercase,
    )

    # set a normalizer if requested
    if normalization == "nfc":
        # set the normalizer
        tokenizer.normalizer = NFC()
    elif normalization == "nfd":
        # set the normalizer
        tokenizer.normalizer = NFD()
    elif normalization == "nfkc":
        # set the normalizer
        tokenizer.normalizer = NFKC()
    elif normalization == "nfkd":
        # set the normalizer
        tokenizer.normalizer = NFKD()

    # add special tokens
    tokenizer.add_special_tokens(special_tokens)

    # add any
    add_tokens = get_custom_tokens(
        include_whitespace=True,
        include_markdown=True,
        include_html=True,
        include_json=True,
        include_xml=True,
        include_years=True,
        include_citations=True,
        lowercase=lowercase,
    )

    # now train from the generator
    print("Training tokenizer.")
    tokenizer.train_from_iterator(
        yield_field_from_json(input_path_list),
        vocab_size=vocab_size - len(special_tokens) - len(add_tokens),
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    # add any that weren't already present
    print("Adding custom tokens.")
    extend_tokens = []
    original_vocab = tokenizer.get_vocab()
    for token in add_tokens:
        if token not in original_vocab:
            extend_tokens.append(token)
    tokenizer.add_tokens(extend_tokens)

    # pad up to the next power of 2 if requested
    new_tokens = []
    if pad2:
        # current vocab power of 2
        current_vocab_size = log2(tokenizer.get_vocab_size())

        # round up to next integer power of 2
        target_vocab_size = 2 ** (int(current_vocab_size) + 1)

        # find the longest consecutive space length in the vocab
        new_tokens_needed = target_vocab_size - tokenizer.get_vocab_size()
        current_vocab = tokenizer.get_vocab()
        for sequence_length in range(2, target_vocab_size):
            # add both " " and "\n" * sequence_length if not present
            if " " * sequence_length not in current_vocab:
                new_tokens.append(" " * sequence_length)

            if len(new_tokens) == new_tokens_needed:
                break

            if "\n" * sequence_length not in current_vocab:
                new_tokens.append("\n" * sequence_length)

            if len(new_tokens) == new_tokens_needed:
                break

        # add the space tokens
        print("Adding power-of-2 padding tokens.")
        tokenizer.add_tokens(new_tokens)
        print(f"Padded vocab to {tokenizer.get_vocab_size()} tokens.")

    # set padding direction
    tokenizer.enable_padding("left", pad_token="<|pad|>")

    # save the tokenizer in both tokenizers and transformers formats
    tokenizer.save_model(str(output_path.absolute()))
    tokenizer.save(str(output_path.absolute() / "tokenizer.json"))
    with open(
        str(output_path.absolute() / "tokenizer_config.json"), "wt", encoding="utf-8"
    ) as tokenizer_config_file:
        tokenizer_config_file.write(
            json.dumps(
                {
                    "unk_token": "<|unk|>",
                    "bos_token": "<|start|>",
                    "eos_token": "<|end|>",
                    "pad_token": "<|pad|>",
                    "add_prefix_space": False,
                    "tokenizer_class": "GPTNeoXTokenizer",
                }
            )
        )

    # end time
    end_time = time.time()

    # output stored location, size, special token count, custom token count, and pad token count
    print(f"Special tokens: {len(special_tokens)}")
    print(f"Custom tokens: {len(extend_tokens)}")
    print(f"Power-of-2 pad tokens: {len(new_tokens)}")
    print(f"Final vocab size: {tokenizer.get_vocab_size()}")
    print(f"Training time: {(end_time - start_time):.2f} seconds")
    print(f"Output path: {output_path.absolute()}")


if __name__ == "__main__":
    # set up the argparser
    parser = argparse.ArgumentParser(description="Train a GPT NeoX tokenizer.")

    # add arguments
    parser.add_argument(
        "input_path",
        # path or list of paths separated by commas
        type=str,
        help="Path to the input JSONL file(s)",
    )
    parser.add_argument("output_path", type=str, help="Path to the output directory")
    parser.add_argument(
        "--vocab_size", type=int, default=32768, help="Size of the vocabulary"
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=1,
        help="Minimum frequency of a token to be included in the vocabulary",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="nfkc",
        help="Normalization to apply to the text",
    )
    parser.add_argument(
        "--lowercase", action="store_true", help="Whether to lowercase the text"
    )

    # pad to 2**x
    parser.add_argument(
        "--pad2",
        action="store_true",
        help="Whether to pad the vocab to the next power of 2",
    )

    # parse
    args = parser.parse_args()

    # get the input path
    arg_input_paths = args.input_path.split(",")
    arg_input_paths = [Path(input_path) for input_path in arg_input_paths]

    # get the output path
    arg_output_path = Path(args.output_path)
    arg_output_path.mkdir(parents=True, exist_ok=True)

    # train the tokenizer
    train_tokenizer(
        input_path_list=arg_input_paths,
        output_path=arg_output_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        normalization=args.normalization,
        lowercase=args.lowercase,
        pad2=args.pad2,
    )
