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
            # "./kl3m-001-32k",
            # "./kl3m-003-64k",
            # "./kl3m-004-128k-uncased",
            # "./kl3m-004-128k-uncased-mlm",
            # "./kl3m-004-128k-cased",
            # "kl3m-004-char-4k-cased",
            "kl3m-004-char-8k-cased",
            "kl3m-004-char-16k-cased",
        )
    ]

    # create the huggingface API
    hf_api = HfApi()

    # load tokenizers
    for path in PATH_LIST:
        # get the name from the last component
        tokenizer_name = path.split("/")[-1]

        # load tokenizer
        repo_id = f"alea-institute/{tokenizer_name}"

        # use HfApi to upload all files from that folder
        hf_api = HfApi()
        try:
            repo_url = hf_api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=True,
            )
        except Exception as e:
            print(f"Error creating repo {repo_id}: {e}")

        # now upload all files to main
        hf_api.upload_folder(
            repo_id=repo_id,
            folder_path=path,
        )
