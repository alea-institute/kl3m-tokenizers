"""
Compare kl3m-001, kl3m-003, and gpt-4o tokenizers.
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
    kl3m_001 = AutoTokenizer.from_pretrained("alea-institute/kl3m-001-32k")
    kl3m_003 = AutoTokenizer.from_pretrained("alea-institute/kl3m-003-64k")
    kl3m_004_uncased = AutoTokenizer.from_pretrained("kl3m-004-128k-uncased")
    gpt_4o = tiktoken.encoding_for_model("gpt-4o")

    # input loop
    while True:
        try:
            # get the input
            text = input("Enter text to tokenize (q to quit): ")
            if text.lower() == "q":
                break

            # tokenize
            kl3m_001_encoded = kl3m_001(text)
            kl3m_003_encoded = kl3m_003(text)
            kl3m_004_uncased_encoded = kl3m_004_uncased(text)
            gpt_4o_encoded = gpt_4o.encode(text)

            # get human-readable tokens
            kl3m_001_tokens = kl3m_001.convert_ids_to_tokens(
                kl3m_001_encoded["input_ids"]
            )
            kl3m_003_tokens = kl3m_003.convert_ids_to_tokens(
                kl3m_003_encoded["input_ids"]
            )
            kl3m_004_uncased_tokens = kl3m_004_uncased.convert_ids_to_tokens(
                kl3m_004_uncased_encoded["input_ids"]
            )
            gpt_4o_tokens = [gpt_4o.decode([token]) for token in gpt_4o_encoded]

            # pretty formatting for these so they are as easy to inspect as possible
            print_token_sequences(
                {
                    "kl3m-001-32k": kl3m_001_tokens,
                    "kl3m-003-64k": kl3m_003_tokens,
                    "kl3m-004-128k-uncased": kl3m_004_uncased_tokens,
                    "gpt-4o": gpt_4o_tokens,
                }
            )
        except KeyboardInterrupt:
            break
