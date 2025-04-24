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
import statistics
import collections
from math import log2
from pathlib import Path
from typing import Generator, Optional, List, Dict, Any, Tuple, Deque

# packages
import httpx
import datasets
from cheesecloth import char_entropy, unigram_entropy
from tokenizers import Tokenizer, normalizers
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, NFD, NFKC, NFKD, Lowercase, Sequence
from tokenizers.pre_tokenizers import RandomWhitespaceSplit, RandomChunkSplit
from transformers import AutoTokenizer


# Entropy filter class to manage running entropy stats
class EntropyFilter:
    """
    A filter that tracks entropy statistics and filters samples based on percentiles.
    
    This class maintains a queue of recent entropy values and can determine 
    if a new sample's entropy is within acceptable percentile bounds.
    Uses Python's built-in statistics module instead of numpy for efficiency.
    """
    
    def __init__(self, threshold: float, queue_size: int = 100000):
        """
        Initialize the entropy filter.
        
        Args:
            threshold: The percentile threshold (0.0-0.5). Values in the bottom or top 
                       threshold percentiles will be filtered out.
            queue_size: The maximum number of historical entropy values to maintain.
        """
        self.threshold = threshold
        self.queue_size = queue_size
        self.char_entropies: Deque[float] = collections.deque(maxlen=queue_size)
        self.unigram_entropies: Deque[float] = collections.deque(maxlen=queue_size)
        
        # Cache for percentile boundaries, updated periodically
        self._cached_boundaries = {
            'char_low': None,
            'char_high': None,
            'unigram_low': None,
            'unigram_high': None,
            'last_update_count': 0
        }
        # Only recalculate percentiles every N samples to improve performance
        self._recalc_interval = 1000
        
    def update(self, text: str) -> None:
        """
        Update the filter with a new text sample's entropy values.
        
        Args:
            text: The text sample to compute entropy for.
        """
        if not text or len(text) < 10:  # Skip very short texts
            return
        
        try:
            # Calculate entropy values
            c_entropy = char_entropy(text)
            u_entropy = unigram_entropy(text, True, True)
            
            # Add to queues
            self.char_entropies.append(c_entropy)
            self.unigram_entropies.append(u_entropy)
            
            # Update cached boundaries if needed
            self._maybe_update_boundaries()
            
        except Exception as e:
            print(f"Error calculating entropy: {e}")
    
    def _maybe_update_boundaries(self) -> None:
        """
        Update the cached percentile boundaries if enough new samples have been added.
        This improves performance by avoiding recalculating percentiles for every sample.
        """
        current_count = len(self.char_entropies)
        
        # If this is the first time or we've added enough new samples since last update
        if (self._cached_boundaries['last_update_count'] == 0 or 
            current_count - self._cached_boundaries['last_update_count'] >= self._recalc_interval):
            
            # Only calculate if we have enough samples
            if current_count >= 1000:
                try:
                    # Calculate the quantile points for low and high thresholds
                    low_quantile = self.threshold
                    high_quantile = 1.0 - self.threshold
                    
                    # Get a sorted copy of the deques for calculating quantiles
                    # This is more efficient than converting to numpy arrays
                    sorted_char = sorted(self.char_entropies)
                    sorted_unigram = sorted(self.unigram_entropies)
                    
                    # Calculate percentile boundaries using statistics module
                    self._cached_boundaries['char_low'] = statistics.quantiles(
                        sorted_char, n=100, method='inclusive'
                    )[int(self.threshold * 100) - 1]
                    
                    self._cached_boundaries['char_high'] = statistics.quantiles(
                        sorted_char, n=100, method='inclusive'
                    )[int((1 - self.threshold) * 100) - 1]
                    
                    self._cached_boundaries['unigram_low'] = statistics.quantiles(
                        sorted_unigram, n=100, method='inclusive'
                    )[int(self.threshold * 100) - 1]
                    
                    self._cached_boundaries['unigram_high'] = statistics.quantiles(
                        sorted_unigram, n=100, method='inclusive'
                    )[int((1 - self.threshold) * 100) - 1]
                    
                    # Update the last update count
                    self._cached_boundaries['last_update_count'] = current_count
                    
                except Exception as e:
                    print(f"Error updating percentile boundaries: {e}")
    
    def should_keep(self, text: str) -> bool:
        """
        Determine if a text sample should be kept based on its entropy.
        
        Args:
            text: The text sample to evaluate.
            
        Returns:
            bool: True if the sample should be kept, False if it should be filtered out.
        """
        # If we don't have enough samples yet, keep everything
        if len(self.char_entropies) < 1000:
            self.update(text)  # Still update our stats
            return True
        
        try:
            # Calculate entropy values for this sample
            c_entropy = char_entropy(text)
            u_entropy = unigram_entropy(text, True, True)
            
            # Get cached percentile boundaries
            char_low = self._cached_boundaries['char_low']
            char_high = self._cached_boundaries['char_high']
            unigram_low = self._cached_boundaries['unigram_low']
            unigram_high = self._cached_boundaries['unigram_high']
            
            # Safety check - if boundaries are not calculated yet, calculate them now
            if any(bound is None for bound in [char_low, char_high, unigram_low, unigram_high]):
                self._maybe_update_boundaries()
                # Get the updated values
                char_low = self._cached_boundaries['char_low']
                char_high = self._cached_boundaries['char_high']
                unigram_low = self._cached_boundaries['unigram_low']
                unigram_high = self._cached_boundaries['unigram_high']
            
            # Decide whether to keep based on both entropy metrics
            char_ok = char_low <= c_entropy <= char_high
            unigram_ok = unigram_low <= u_entropy <= unigram_high
            
            # Only if we keep it do we update our stats
            if char_ok and unigram_ok:
                self.update(text)
                return True
            
            return False
            
        except Exception as e:
            print(f"Error evaluating entropy: {e}")
            return True  # On error, default to keeping the sample


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
    dataset_sources: List[Dict[str, Any]],
    input_tokenizer_name: str,
    sample_rate: Optional[float] = None,  # Global sample rate (as fallback)
    lowercase: bool = False,
    shuffle_seed: Optional[int] = None,
    batch_size: int = 100,
    filter_entropy: Optional[float] = None,  # Entropy filtering threshold
) -> Generator[str, None, None]:
    """
    Yield text from multiple datasets or local files, decoding tokens using the input tokenizer.
    Uses batch decoding for improved performance and can filter based on text entropy.

    Args:
        dataset_sources (List[Dict[str, Any]]): List of dataset sources, each containing:
            - 'source': str - Either 'huggingface' or 'local'
            - For huggingface: 'name': str - Dataset name in format "name" or "name/config"
            - For local: 'path': str - Path to local file (.txt, .jsonl, .jsonl.gz)
                       'field': Optional[str] - For jsonl files, the field containing text or tokens
            - 'sample_rate': Optional[float] - Source-specific sampling rate (overrides global)
        input_tokenizer_name (str): Name or path of the tokenizer to use for decoding tokens
        sample_rate (Optional[float]): Global sampling rate (used if source doesn't specify its own)
        lowercase (bool): Whether to lowercase the text
        shuffle_seed (Optional[int]): If provided, shuffle with this random seed
        batch_size (int): Size of batches to load from datasets and to decode
        filter_entropy (Optional[float]): If provided, filter samples based on character and unigram
                                        entropy. Value should be between 0.0 and 0.5, representing
                                        the percentile thresholds to exclude (e.g., 0.1 means exclude
                                        samples below 10th percentile or above 90th percentile).

    Yields:
        Generator[str, None, None]: Generator of decoded text
    """
    # Load the input tokenizer
    print(f"Loading input tokenizer: {input_tokenizer_name}")
    input_tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_name)
    
    # Set up RNG for sampling
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else random.Random()
    
    # Set up entropy filter if needed
    entropy_filter = None
    if filter_entropy is not None:
        if 0 < filter_entropy < 0.5:
            print(f"Using entropy filter with threshold: {filter_entropy}")
            entropy_filter = EntropyFilter(filter_entropy)
        else:
            print(f"Warning: Invalid filter_entropy value {filter_entropy}. Must be between 0 and 0.5. Disabling entropy filtering.")
    
    # Process each dataset source
    for source_config in dataset_sources:
        # Get source-specific sample rate (or use global rate as fallback)
        source_sample_rate = source_config.get('sample_rate', sample_rate)
        source_type = source_config.get('source', 'huggingface')
        
        if source_type == 'huggingface':
            # Process HuggingFace dataset
            dataset_name = source_config['name']
            try:
                print(f"Loading HuggingFace dataset: {dataset_name}")
                print(f"Using sample rate: {source_sample_rate}")
                
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
                        
                        # Apply sampling if requested
                        token_batches = []
                        
                        for tokens in batch["tokens"]:
                            if source_sample_rate is None or rng.random() <= source_sample_rate:
                                token_batches.append(tokens)
                        
                        if not token_batches:
                            continue
                        
                        # Batch decode for better performance
                        try:
                            decoded_texts = input_tokenizer.batch_decode(token_batches)
                            
                            # Process each decoded text
                            for text in decoded_texts:
                                # Apply lowercase if requested
                                if lowercase:
                                    text = text.lower()
                                
                                # Apply entropy filtering if configured
                                if entropy_filter is not None:
                                    if entropy_filter.should_keep(text):
                                        yield text
                                else:
                                    yield text
                                
                        except Exception as e:
                            print(f"Error batch decoding tokens: {e}")
                            
                            # Fall back to individual decoding if batch fails
                            for tokens in token_batches:
                                try:
                                    text = input_tokenizer.decode(tokens)
                                    
                                    # Apply lowercase if requested
                                    if lowercase:
                                        text = text.lower()
                                    
                                    # Apply entropy filtering if configured
                                    if entropy_filter is not None:
                                        if entropy_filter.should_keep(text):
                                            yield text
                                    else:
                                        yield text
                                    
                                except Exception as e:
                                    print(f"Error decoding tokens: {e}")
                                    continue
                    
                    print(f"Finished processing {split_name} from {dataset_name}")
                    
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                continue
        
        elif source_type == 'local':
            # Process local file
            file_path = source_config['path']
            field = source_config.get('field', None)  # For JSON files
            has_tokens = source_config.get('has_tokens', False)  # Whether file contains token IDs
            
            try:
                print(f"Loading local file: {file_path}")
                print(f"Using sample rate: {source_sample_rate}")

                extension = Path(file_path).suffix.lower()
                if extension == '.gz':
                    opener = gzip.open
                else:
                    opener = open

                with opener(file_path, 'rt', encoding='utf-8') as input_file:
                    for line in input_file:
                        # apply sampling
                        if source_sample_rate is not None and rng.random() > source_sample_rate:
                            continue

                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON line: {line}")
                            continue

                        # get the text or tokens based on the field
                        field_value = record.get(field, None)
                        if field_value is None:
                            print(f"Field '{field}' not found in record: {record}")
                            continue

                        # check if text or tokens
                        if has_tokens:
                            # decode tokens
                            if isinstance(field_value, list):
                                tokens = field_value
                            else:
                                print(f"Expected list of tokens, got: {field_value}")
                                continue

                            # Decode the tokens
                            text = input_tokenizer.decode(tokens)
                        else:
                            # Assume it's text
                            text = field_value

                        # Apply lowercase if requested
                        if lowercase:
                            text = text.lower()

                        # Apply entropy filtering if configured
                        if entropy_filter is not None:
                            if entropy_filter.should_keep(text):
                                yield text
                        else:
                            yield text

                print(f"Finished processing local file: {file_path}")
            
            except Exception as e:
                print(f"Error processing local file {file_path}: {e}")
                continue
        
        else:
            print(f"Unsupported source type: {source_type}")
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
    
    # Dataset sources can be in old format (list of names) or new format (list of configs)
    if "datasets" in config and isinstance(config["datasets"], list):
        # Check if it's the old format (list of dataset names)
        if all(isinstance(item, str) for item in config["datasets"]):
            # Convert to new format
            dataset_sources = [{"source": "huggingface", "name": name} for name in config["datasets"]]
        else:
            # Already in new format
            dataset_sources = config["datasets"]
    elif "dataset_sources" in config:
        # Direct use of new format
        dataset_sources = config["dataset_sources"]
    else:
        # Neither format found
        raise ValueError("Config must contain either 'datasets' (list of names) or 'dataset_sources' (list of configs)")
    
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
    filter_entropy = config.get("filter_entropy", None)
    prune_min_frequency = config.get("prune_min_frequency", 10)
    prune_step_interval = config.get("prune_step_interval", 10000)
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
        
        # Prepare trainer kwargs, only adding pruning params if they're non-null
        trainer_kwargs = {
            "vocab_size": vocab_size - len(special_tokens) - len(add_tokens),
            "min_frequency": min_frequency,
            "special_tokens": special_tokens,
            "show_progress": True
        }
        
        # Only add pruning parameters if they are not None
        if prune_min_frequency is not None:
            trainer_kwargs["prune_min_frequency"] = prune_min_frequency
        if prune_step_interval is not None:
            trainer_kwargs["prune_step_interval"] = prune_step_interval
            
        # Initialize the trainer with the kwargs
        trainer = trainers.BpeTrainer(**trainer_kwargs)
        
        # Use our dataset generator
        print(f"Training tokenizer with {pretokenizer_type} pretokenizer using iterator.")
        tokenizer.train_from_iterator(
            yield_text_from_datasets(
                dataset_sources=dataset_sources,
                input_tokenizer_name=input_tokenizer,
                sample_rate=sample_rate,  # This is now used as a fallback if a source doesn't specify its own
                lowercase=lowercase,
                shuffle_seed=shuffle_seed,
                batch_size=batch_size,
                filter_entropy=filter_entropy
            ),
            trainer=trainer
        )
    else:
        # For ByteLevelBPETokenizer, we use train_from_iterator
        print("Training tokenizer without pretokenizer.")
        tokenizer.train_from_iterator(
            yield_text_from_datasets(
                dataset_sources=dataset_sources,
                input_tokenizer_name=input_tokenizer,
                sample_rate=sample_rate,  # This is now used as a fallback if a source doesn't specify its own
                lowercase=lowercase,
                shuffle_seed=shuffle_seed,
                batch_size=batch_size,
                filter_entropy=filter_entropy
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
        print(f"  Prune min frequency: {prune_min_frequency}")
        print(f"  Prune step interval: {prune_step_interval}")
    
    if filter_entropy is not None:
        print(f"Entropy filtering threshold: {filter_entropy} (excluding top and bottom {filter_entropy * 100:.1f}% of samples)")


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