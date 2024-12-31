"""
Compare the ALEA character tokenizers with each other.
"""

# imports

# packages
import tiktoken
from transformers import AutoTokenizer


def print_token_sequences(sequences: dict[str, list[str]], padding: int = 1):
    """
    Print running columns of token sequences for easy comparison with improved formatting.

    Args:
        sequences (dict[str, list[str]]): A mapping of tokenizer names to token sequences.
        padding (int): Number of spaces to add as padding on each side of cell content.

    Returns:
        None
    """
    # get the keys and max widths
    keys = sorted(list(sequences.keys()))
    max_widths = [
        max(len(key), max(len(token) for token in sequences[key])) for key in keys
    ]
    cell_widths = [width + 2 * padding for width in max_widths]

    # helper functions for rows and horizontal rules
    def print_row(row_data):
        print("|", end="")
        for i, item in enumerate(row_data):
            print(f" {item:{cell_widths[i]}} |", end="")
        print()

    def print_separator(char="-"):
        print(f"+{'+'.join([char * (width + 2) for width in cell_widths])}+")

    # headers
    print_separator("=")
    print_row(keys)
    print_separator("=")

    # ge the dim
    max_rows = max(len(sequence) for sequence in sequences.values())
    for row in range(max_rows):
        row_data = []
        for key in keys:
            token = sequences[key][row] if row < len(sequences[key]) else ""
            row_data.append(token)
        print_row(row_data)
        if row < max_rows - 1:
            print_separator()

    # footer
    print_separator("=")


if __name__ == "__main__":
    # get the tokenizers
    kl3m_4k = AutoTokenizer.from_pretrained("alea-institute/kl3m-004-char-4k-cased")
    kl3m_8k = AutoTokenizer.from_pretrained("alea-institute/kl3m-004-char-8k-cased")
    kl3m_16k = AutoTokenizer.from_pretrained("alea-institute/kl3m-004-char-16k-cased")

    # input loop
    while True:
        try:
            # get the input
            text = input("Enter text to tokenize (q to quit): ")
            if text.lower() == "q":
                break

            # tokenize
            kl3m_4k_encoded = kl3m_4k(text)
            kl3m_8k_encoded = kl3m_8k(text)
            kl3m_16k_encoded = kl3m_16k(text)

            # get human-readable tokens
            kl3m_4k_tokens = kl3m_4k.convert_ids_to_tokens(kl3m_4k_encoded["input_ids"])
            kl3m_8k_tokens = kl3m_8k.convert_ids_to_tokens(kl3m_8k_encoded["input_ids"])
            kl3m_16k_tokens = kl3m_16k.convert_ids_to_tokens(kl3m_16k_encoded["input_ids"])

            # pretty formatting for these so they are as easy to inspect as possible
            print_token_sequences(
                {
                    "0 kl3m-004-4k-cased": kl3m_4k_tokens,
                    "1 kl3m-004-8k-cased": kl3m_8k_tokens,
                    "2 kl3m-004-16k-cased": kl3m_16k_tokens,
                }
            )
        except KeyboardInterrupt:
            break
