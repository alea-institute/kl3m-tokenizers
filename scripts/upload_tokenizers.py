"""
Load tokenizers and upload to HF Hub with relevant metadata.
"""

# ignore order for HF token path
# pylint: disable=wrong-import-position

# imports
import os
from pathlib import Path

# set hf token path to ../.hftoken
os.environ["HF_TOKEN_PATH"] = (
    (Path(__file__).parent.parent / ".hftoken").resolve().as_posix()
)


# packages
from huggingface_hub import HfApi
from transformers import AutoTokenizer

if __name__ == "__main__":
    # get paths
    PATH_LIST: list[str] = [
        # this is just getting the absolute path for `../kl3m-embedding-00*` as a string
        (Path(__file__).parent.parent / p).resolve().as_posix()
        for p in (
            "./kl3m-001-32k",
            "./kl3m-003-64k",
            "./kl3m-004-128k-uncased",
            "./kl3m-004-128k-cased",
        )
    ]

    # create the huggingface API
    hf_api = HfApi()

    # load tokenizers
    for path in PATH_LIST:
        # get the name from the last component
        tokenizer_name = path.split("/")[-1]

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.push_to_hub(
            repo_id=f"alea-institute/{tokenizer_name}", revision="main", private=True
        )

        # get the README.md an dupload it to the repo
        readme_path = Path(path) / "README.md"
        if readme_path.exists():
            hf_api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=f"alea-institute/{tokenizer_name}",
            )
        else:
            print(f"README.md not found for {tokenizer_name}")
