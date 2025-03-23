"""
Train a BPE tokenizer for KL3M-005 with support for random whitespace and random chunk pretokenizers,
using a configuration file to specify input tokenizer and datasets.
"""

# imports
import argparse
import json
import gzip
import random
import string
import time
from math import log2
from pathlib import Path
from typing import Generator, Optional, List, Dict, Any

# packages
import httpx
import datasets
from tokenizers import Tokenizer, normalizers
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, NFD, NFKC, NFKD, Lowercase, Sequence
from tokenizers.pre_tokenizers import RandomWhitespaceSplit, RandomChunkSplit
from transformers import AutoTokenizer


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

    # lowercase if requested
    if lowercase:
        tokens = [token.lower() for token in tokens]

    # sort and unique
    tokens = sorted(list(set(tokens)))

    # remove any single-length tokens
    tokens = [token for token in tokens if len(token) > 1]

    # return the tokens
    return tokens


def setup_pretokenizer(
    pretokenizer_type: Optional[str] = None,
    split_probability: float = 0.3,
    min_length: int = 2,
    max_length: int = 5,
) -> Optional[RandomWhitespaceSplit | RandomChunkSplit]:
    """
    Set up the pretokenizer based on the specified type.

    Args:
        pretokenizer_type (Optional[str]): Type of pretokenizer to use ('whitespace', 'chunk', or None)
        split_probability (float): Probability of splitting at whitespace (0.0-1.0)
        min_length (int): Minimum length of chunks for RandomChunkSplit
        max_length (int): Maximum length of chunks for RandomChunkSplit

    Returns:
        Optional[RandomWhitespaceSplit | RandomChunkSplit]: The configured pretokenizer or None
    """
    if pretokenizer_type is None:
        return None
    elif pretokenizer_type.lower() == "whitespace":
        return RandomWhitespaceSplit(split_probability=split_probability)
    elif pretokenizer_type.lower() == "chunk":
        return RandomChunkSplit(min_length=min_length, max_length=max_length)
    else:
        raise ValueError(f"Unsupported pretokenizer type: {pretokenizer_type}")


def yield_text_from_datasets(
    dataset_names: List[str],
    input_tokenizer_name: str,
    sample_rate: Optional[float] = None,
    lowercase: bool = False,
    shuffle_seed: Optional[int] = None,
    batch_size: int = 1000,
) -> Generator[str, None, None]:
    """
    Yield text from multiple datasets, decoding tokens using the input tokenizer.

    Args:
        dataset_names (List[str]): List of HuggingFace dataset names in format "name" or "name/config"
        input_tokenizer_name (str): Name or path of the tokenizer to use for decoding tokens
        sample_rate (Optional[float]): If provided, randomly sample this fraction of examples
        lowercase (bool): Whether to lowercase the text
        shuffle_seed (Optional[int]): If provided, shuffle with this random seed
        batch_size (int): Size of batches to load from datasets

    Yields:
        Generator[str, None, None]: Generator of decoded text
    """
    # Load the input tokenizer
    print(f"Loading input tokenizer: {input_tokenizer_name}")
    input_tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_name)
    
    # Set up RNG for sampling
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else random.Random()
    
    # Process each dataset
    for dataset_name in dataset_names:
        try:
            print(f"Loading dataset: {dataset_name}")
            
            # Handle dataset with config vs without config
            dataset = datasets.load_dataset(dataset_name)

            # Process each split in the dataset (train, validation, test)
            for split_name in dataset:
                split = dataset[split_name]
                print(f"Processing split: {split_name} with {len(split)} examples")
                
                # Process in batches
                for i in range(0, len(split), batch_size):
                    batch = split[i:i+batch_size]
                    
                    # Assuming 'tokens' field exists in the dataset
                    if "tokens" not in batch:
                        print(f"Warning: 'tokens' field not found in {dataset_name}/{split_name}, skipping")
                        continue
                    
                    for tokens in batch["tokens"]:
                        # Apply sampling if requested
                        if sample_rate is not None and rng.random() > sample_rate:
                            continue
                        
                        # Decode tokens to text
                        try:
                            text = input_tokenizer.decode(tokens)
                            
                            # Apply lowercase if requested
                            if lowercase:
                                text = text.lower()
                                
                            yield text
                            
                        except Exception as e:
                            print(f"Error decoding tokens: {e}")
                            continue
                
                print(f"Finished processing {split_name} from {dataset_name}")
                
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue


def train_tokenizer(
    config_path: Path,
    output_path: Path,
):
    """
    Train a tokenizer based on configuration file.

    Args:
        config_path (Path): Path to the configuration file
        output_path (Path): Path to the output tokenizer directory
    """
    # Start time
    start_time = time.time()
    
    # Load config
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    
    # Extract config parameters with defaults
    input_tokenizer = config["input_tokenizer"]
    dataset_names = config["datasets"]
    vocab_size = config.get("vocab_size", 32768)
    min_frequency = config.get("min_frequency", 10)
    normalization = config.get("normalization", "nfkc")
    lowercase = config.get("lowercase", False)
    pad2 = config.get("pad2", False)
    pretokenizer_type = config.get("pretokenizer_type", None)
    split_probability = config.get("split_probability", 0.3)
    min_length = config.get("min_length", 2)
    max_length = config.get("max_length", 5)
    sample_rate = config.get("sample_rate", None)
    shuffle_seed = config.get("shuffle_seed", None)
    batch_size = config.get("batch_size", 1000)
    custom_token_settings = config.get("custom_tokens", {
        "include_whitespace": True,
        "include_markdown": True,
        "include_html": True,
        "include_json": True,
        "include_xml": True,
        "include_years": True,
        "include_citations": True
    })

    # Setup the special token list
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

    # Check if we're using a pretokenizer
    if pretokenizer_type:
        # Initialize a tokenizer with BPE model
        tokenizer = Tokenizer(BPE())
        
        # Set up the pretokenizer
        pretokenizer = setup_pretokenizer(
            pretokenizer_type=pretokenizer_type,
            split_probability=split_probability,
            min_length=min_length,
            max_length=max_length
        )
        tokenizer.pre_tokenizer = pretokenizer
        
        # Set up normalizer
        normalizer_sequence = []
        
        if normalization == "nfc":
            normalizer_sequence.append(NFC())
        elif normalization == "nfd":
            normalizer_sequence.append(NFD())
        elif normalization == "nfkc":
            normalizer_sequence.append(NFKC())
        elif normalization == "nfkd":
            normalizer_sequence.append(NFKD())
            
        if lowercase:
            normalizer_sequence.append(Lowercase())
            
        if normalizer_sequence:
            tokenizer.normalizer = Sequence(normalizer_sequence)
    else:
        # Use ByteLevelBPETokenizer for standard training
        tokenizer = ByteLevelBPETokenizer(
            lowercase=lowercase,
        )
        
        # set a normalizer if requested
        normalizers_list = []

        if normalization == "nfc":
            normalizers_list.append(NFC())
        elif normalization == "nfd":
            normalizers_list.append(NFD())
        elif normalization == "nfkc":
            normalizers_list.append(NFKC())
        elif normalization == "nfkd":
            normalizers_list.append(NFKD())

        # add lowercase if requested
        if lowercase:
            normalizers_list.append(Lowercase())

        # set normalizers from sequence
        if normalizers_list:
            tokenizer.normalizer = Sequence(normalizers_list)

    # add special tokens
    tokenizer.add_special_tokens(special_tokens)

    # add custom tokens
    add_tokens = get_custom_tokens(
        **custom_token_settings,
        lowercase=lowercase,
    )

    # Define the training method based on tokenizer type
    if pretokenizer_type:
        print(f"Training tokenizer with {pretokenizer_type} pretokenizer.")
        # For custom tokenizer with pre-tokenizer, we use trainers.BpeTrainer
        from tokenizers import trainers
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size - len(special_tokens) - len(add_tokens),
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True
        )
        
        # Use our dataset generator
        print(f"Training tokenizer with {pretokenizer_type} pretokenizer using iterator.")
        tokenizer.train_from_iterator(
            yield_text_from_datasets(
                dataset_names=dataset_names,
                input_tokenizer_name=input_tokenizer,
                sample_rate=sample_rate,
                lowercase=lowercase,
                shuffle_seed=shuffle_seed,
                batch_size=batch_size
            ),
            trainer=trainer
        )
    else:
        # For ByteLevelBPETokenizer, we use train_from_iterator
        print("Training tokenizer without pretokenizer.")
        tokenizer.train_from_iterator(
            yield_text_from_datasets(
                dataset_names=dataset_names,
                input_tokenizer_name=input_tokenizer,
                sample_rate=sample_rate,
                lowercase=lowercase,
                shuffle_seed=shuffle_seed,
                batch_size=batch_size
            ),
            vocab_size=vocab_size - len(special_tokens) - len(add_tokens),
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )

    # add any custom tokens that weren't already present
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

    # Set padding direction
    if hasattr(tokenizer, "enable_padding"):
        pad_id = tokenizer.token_to_id("<|pad|>") if hasattr(tokenizer, "token_to_id") else None
        
        # The interface differs based on tokenizer type
        if pretokenizer_type:
            # For the Tokenizer class with pretokenizers
            tokenizer.enable_padding(pad_id=pad_id, pad_token="<|pad|>", direction="left")
        else:
            # For ByteLevelBPETokenizer
            tokenizer.enable_padding(
                "left", pad_id=pad_id, pad_token="<|pad|>"
            )

    # First, save the complete tokenizer (with pretokenizer if used) as a backup
    if hasattr(tokenizer, "save_model"):
        tokenizer.save_model(str(output_path.absolute()))
    
    if pretokenizer_type:
        # Save the full version with pretokenizer to a backup file for training
        training_tokenizer_path = str(output_path.absolute() / "tokenizer_with_pretokenizer.json")
        tokenizer.save(training_tokenizer_path)
        print(f"Training tokenizer with pretokenizer saved to: {training_tokenizer_path}")
        
        # For the main tokenizer.json, save a version without the pretokenizer
        print("Creating inference version without pretokenizer for tokenizer.json")
        temp_tokenizer_path = str(output_path.absolute() / "temp_tokenizer.json")
        tokenizer.save(temp_tokenizer_path)
        
        # Read the JSON
        with open(temp_tokenizer_path, "r") as f:
            tokenizer_data = json.load(f)
        
        # Remove pre-tokenizer field if present
        if "pre_tokenizer" in tokenizer_data:
            del tokenizer_data["pre_tokenizer"]
        
        # Write modified tokenizer as the main tokenizer.json
        main_tokenizer_path = str(output_path.absolute() / "tokenizer.json")
        with open(main_tokenizer_path, "w") as f:
            json.dump(tokenizer_data, f, indent=2)
        
        # Clean up temp file
        import os
        os.remove(temp_tokenizer_path)
        
        print(f"Inference-ready tokenizer (no pretokenizer) saved as main tokenizer.json")
    else:
        # Standard save without pretokenizer
        tokenizer.save(str(output_path.absolute() / "tokenizer.json"))
    
    # Save the tokenizer config
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
                    "pretokenizer_type": pretokenizer_type,
                }
            )
        )

    # Save the training config for reference
    with open(
        str(output_path.absolute() / "training_config.json"), "wt", encoding="utf-8"
    ) as training_config_file:
        training_config_file.write(json.dumps(config, indent=2))

    # end time
    end_time = time.time()

    # output stored location, size, special token count, custom token count, and pad token count
    print(f"Special tokens: {len(special_tokens)}")
    print(f"Custom tokens: {len(extend_tokens)}")
    print(f"Power-of-2 pad tokens: {len(new_tokens)}")
    print(f"Final vocab size: {tokenizer.get_vocab_size()}")
    print(f"Training time: {(end_time - start_time):.2f} seconds")
    print(f"Output path: {output_path.absolute()}")
    if pretokenizer_type:
        print(f"Pretokenizer: {pretokenizer_type}")
        if pretokenizer_type.lower() == "whitespace":
            print(f"  Split probability: {split_probability}")
        elif pretokenizer_type.lower() == "chunk":
            print(f"  Min length: {min_length}, Max length: {max_length}")


if __name__ == "__main__":
    # set up the argparser
    parser = argparse.ArgumentParser(description="Train a tokenizer from a configuration file.")

    # Add arguments
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the configuration JSON file",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the output directory",
    )

    # Parse arguments
    args = parser.parse_args()

    # Get the config path
    config_path = Path(args.config_path)

    # Get the output path
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Train the tokenizer
    train_tokenizer(
        config_path=config_path,
        output_path=output_path,
    )