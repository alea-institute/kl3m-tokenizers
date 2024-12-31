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
from tokenizers.normalizers import NFC, NFD, NFKC, NFKD, Lowercase, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


# jsonl generator
def yield_field_from_gz(
    input_path_list: list[Path],
    lowercase: bool = False,
) -> Generator[str, None, None]:
    """
    Yield a field from a JSONL file.

    Args:
        input_path_list (list[Path]): List of paths to the input JSONL files
        field_name (str): Name of the field to yield
        lowercase (bool): Whether to lowercase the field

    Yields:
        Generator[str, None, None]: Generator of the field
    """
    # yield the field from the JSONL file
    try:
        for input_path in input_path_list:
            try:
                # check if there is a :%f in the input path
                if ":" in input_path.name:
                    # split it into the file and rate
                    path_tokens = str(input_path).split(":", maxsplit=1)
                    input_path, rate_string = Path(path_tokens[0]), path_tokens[1]
                    try:
                        rate = float(rate_string)
                    except Exception as rate_error:  # pylint: disable=broad-except
                        raise RuntimeError(
                            f"Unable to parse rate from path {input_path}: {rate_error}"
                        ) from rate_error
                else:
                    rate = None

                # get the right opener
                if str(input_path).lower().endswith(".gz"):
                    opener = gzip.open
                else:
                    opener = open  # type: ignore

                # drop orjsonl for encoding issues
                with opener(input_path, "rt", encoding="utf-8") as gzip_file:
                    line_number = 0
                    for line in gzip_file:
                        line_number += 1

                        # check if we are sampling
                        if rate and random.random() > rate:
                            continue

                        # parse the record
                        try:
                            # decode the line
                            yield json.loads(line).get("text")
                        except Exception as json_error:  # pylint: disable=broad-except
                            print(
                                f"Unable to parse line {line_number} from {input_path}: {json_error}"
                            )
                            continue

                print(f"Finished parsing {input_path}: {line_number} records")
            except Exception as json_error:  # pylint: disable=broad-except
                print(f"Unable to parse {input_path}: {json_error}")
                continue
    except KeyboardInterrupt:
        # break on ctrl-c but don't fail
        print("Interrupted by user.")
        return


def train_tokenizer(
    input_path_list: list[Path],
    output_path: Path,
    vocab_size: int = 32768,
    min_frequency: int = 10,
    max_chars: int = 3,
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
    normalizers = []

    if normalization == "nfc":
        # set the normalizer
        normalizers.append(NFC())
    elif normalization == "nfd":
        # set the normalizer
        normalizers.append(NFD())
    elif normalization == "nfkc":
        # set the normalizer
        normalizers.append(NFKC())
    elif normalization == "nfkd":
        # set the normalizer
        normalizers.append(NFKD())

    # add lowercase if requested
    if lowercase:
        normalizers.append(Lowercase())

    # set normalizers from sequence
    tokenizer.normalizer = Sequence(normalizers)

    # add special tokens
    tokenizer.add_special_tokens(special_tokens)

    # set up initial alphabet to include whitespace
    initial_alphabet = [
        chr(i) for i in range(0, 255)
    ]

    # now train using the low-level rust wrappers
    print("Training tokenizer.")
    trainer = BpeTrainer(
        vocab_size=vocab_size - len(special_tokens),
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
        # this needs a +1 to max_chars to account for the internal tokenizers logic
        max_token_length=max_chars + 1,
    )

    tokenizer._tokenizer.train_from_iterator(
        yield_field_from_gz(input_path_list, lowercase=lowercase),
        trainer=trainer,
    )

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
    tokenizer.enable_padding(
        "left", pad_id=tokenizer.token_to_id("<|pad|>"), pad_token="<|pad|>"
    )

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
                    "sep_token": "<|sep|>",
                    "cls_token": "<|cls|>",
                    "mask_token": "<|mask|>",
                    "add_prefix_space": False,
                    "do_lower_case": lowercase,
                    "tokenizer_class": "PreTrainedTokenizerFast",
                }
            )
        )

    # end time
    end_time = time.time()

    # output stored location, size, special token count, custom token count, and pad token count
    print(f"Special tokens: {len(special_tokens)}")
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
        "--max_chars",
        type=int,
        default=3,
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
        max_chars=args.max_chars,
        normalization=args.normalization,
        lowercase=args.lowercase,
        pad2=args.pad2,
    )
