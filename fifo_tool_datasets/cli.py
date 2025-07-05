from __future__ import annotations
import argparse
import os
import re
import shutil
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import cast
import huggingface_hub as hub
# Pylance: suppress missing type stub warning for datasets
from datasets import (  # type: ignore
    concatenate_datasets,
    Dataset
)
from fifo_tool_datasets.sdk.hf_dataset_adapters.common import DatasetAdapter
from fifo_tool_datasets.sdk.hf_dataset_adapters.conversation import ConversationAdapter
from fifo_tool_datasets.sdk.hf_dataset_adapters.sqna import SQNAAdapter
from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import DSLAdapter

ADAPTERS: dict[str, DatasetAdapter] = {
    "conversation": ConversationAdapter(),
    "sqna": SQNAAdapter(),
    "dsl": DSLAdapter()
}

_HF_DATASET_RE = re.compile(r"^[^/]+/[^/]+$")

@dataclass
class DatasetMetadata:
    """
    Represents metadata associated with a downloaded dataset directory.

    Attributes:
        adapter (str | None):
            Name of the adapter (may be None if metadata is missing)

        repo_id (str | None):
            Dataset repository ID on the Hugging Face Hub, or None if not available

        last_download (str | None):
            Timestamp of the last download in 'YYYY-MM-DD HH:MM:SS' format, or None if not available

        sha (str | None):
            Commit SHA of the dataset at the time of download, or None if not available
    """

    adapter: str | None
    repo_id: str | None
    last_download: str | None
    sha: str | None

    @classmethod
    def from_directory(cls, dir_path: str) -> DatasetMetadata:
        """
        Load metadata from a `.hf_meta.json` file in the given directory.
        If the file does not exist or cannot be parsed, returns a DatasetMetadata
        object with all fields set to None.

        Args:
            dir_path (str):
                Path to the directory containing the `.hf_meta.json` file

        Returns:
            DatasetMetadata:
                A metadata object with values from the file or None for missing fields
        """
        path = os.path.join(dir_path, ".hf_meta.json")
        if not os.path.exists(path):
            return cls(None, None, None, None)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                adapter=data.get("adapter"),
                repo_id=data.get("repo_id"),
                last_download=data.get("last_download"),
                sha=data.get("sha"),
            )
        except (OSError, json.JSONDecodeError):
            return cls(None, None, None, None)

    @classmethod
    def from_values(cls, adapter: str, sha: str, repo_id: str) -> DatasetMetadata:
        """
        Create a new metadata object using the current timestamp.

        Args:
            adapter (str):
                Adapter type (must be a known adapter key)

            sha (str):
                Commit SHA of the dataset from the Hugging Face Hub

            repo_id (str):
                Dataset repository ID on the Hugging Face Hub

        Returns:
            DatasetMetadata:
                A populated metadata object
        """
        return cls(
            adapter=adapter,
            repo_id=repo_id,
            last_download=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sha=sha,
        )

    def save(self, dir_path: str) -> None:
        """
        Save this metadata to a `.hf_meta.json` file in the given directory.

        Args:
            dir_path (str):
                Target directory where the metadata file will be written
        """
        path = os.path.join(dir_path, ".hf_meta.json")
        data = {
            "adapter": self.adapter,
            "repo_id": self.repo_id,
            "last_download": self.last_download,
            "sha": self.sha,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

def _is_hf_dataset(name: str) -> bool:
    """
    Check whether a string matches the `username/repo` pattern.

    Args:
        name (str):
            The string to validate.

    Returns:
        bool:
            `True` if `name` looks like a Hugging Face dataset identifier.
    """

    return bool(_HF_DATASET_RE.match(name))


def _count_dat_records(dat_path: str) -> int:
    """
    Return the number of records in a `.dat` file.

    Args:
        dat_path (str):
            Path to the `.dat` file whose records should be counted.

    Returns:
        int:
            Number of records found in the file.
    """
    count = 0
    with open(dat_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "---":
                count += 1
    return max(count - 1, 0)


def _handle_upload(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    meta_data: DatasetMetadata | None=None
) -> None:
    """
    Execute the `upload` command.

    Args:
        args (argparse.Namespace):
            Parsed command-line arguments.

        parser (argparse.ArgumentParser):
            The argument parser used for error reporting.

        meta_data (DatasetMetadata | None):
            Metadata previously loaded from the destination directory. If None,
            the function will attempt to load it from `.hf_meta.json` in the appropriate path.

    The adapter is resolved either from the CLI flag or from `.hf_meta.json` if uploading a
    directory.
    """
    if os.path.isfile(args.src):
        if not args.src.endswith(".dat"):
            parser.error("upload: source file must end with '.dat'")
        src_is_file = True
    elif os.path.isdir(args.src):
        src_is_file = False
    else:
        src_is_file = None
        parser.error(f"upload: source '{args.src}' is not a valid file or directory")

    if not _is_hf_dataset(args.dst):
        parser.error("upload: destination must be in 'username/repo' format")

    if not args.commit_message:
        parser.error("--commit-message is required when uploading to Hugging Face")

    if src_is_file:
        if args.adapter is None:
            parser.error("--adapter is required when uploading a single `.dat` file")
        adapter = ADAPTERS[args.adapter]

        adapter.from_dat_to_hub(
            dat_filename=args.src,
            hub_dataset=args.dst,
            commit_message=args.commit_message,
            seed=args.seed,
            split_ratios=tuple(args.split_ratio),
        )
    else:
        if meta_data is None:
            meta_data = DatasetMetadata.from_directory(args.src)

        adapter_name = args.adapter or meta_data.adapter

        if adapter_name is None:
            parser.error("--adapter is required or must exist in .hf_meta.json")

        adapter = ADAPTERS[adapter_name]

        local_hash = meta_data.sha
        try:
            remote_hash = hub.HfApi().dataset_info(args.dst).sha
        except hub.errors.RepositoryNotFoundError:
            remote_hash = None

        if not args.y:
            if not local_hash:
                parser.error(
                    "Missing .hf_hash: unable to verify remote sync. Use -y to force overwrite."
                )
            if not remote_hash:
                parser.error("Unable to fetch remote hash. Use -y to force overwrite.")
            if local_hash != remote_hash:
                parser.error("Remote dataset has changed. Use -y to overwrite.")

        adapter.from_dir_to_hub(
            dat_dir=args.src,
            hub_dataset=args.dst,
            commit_message=args.commit_message,
        )


def _handle_download(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    meta_data: DatasetMetadata | None=None
) -> None:
    """
    Execute the `download` command.

    Args:
        args (argparse.Namespace):
            Parsed command-line arguments.

        parser (argparse.ArgumentParser):
            The argument parser used for error reporting.

        meta_data (DatasetMetadata | None):
            Metadata previously loaded from the destination directory. If None,
            the function will attempt to load it from `.hf_meta.json` in the appropriate path.
    """
    if not _is_hf_dataset(args.src):
        parser.error("download: source must be in 'username/repo' format")

    dst_is_file = args.dst.endswith(".dat")

    if dst_is_file:
        if os.path.exists(args.dst) and not args.y:
            parser.error(f"Output file '{args.dst}' already exists. Use -y to overwrite.")
    else:
        if os.path.exists(args.dst):
            existing = cast(list[str], os.listdir(args.dst))
            if existing and not args.y:
                parser.error(
                    f"Directory '{args.dst}' already exists and is not empty. Use -y to overwrite."
                )
        os.makedirs(args.dst, exist_ok=True)

    adapter_name = args.adapter

    if dst_is_file:
        if adapter_name is None:
            parser.error("--adapter is required")

        adapter = ADAPTERS[adapter_name]

        dataset_dict = adapter.from_hub_to_dataset_dict(args.src)

        merged = concatenate_datasets([
            dataset_dict["train"],
            dataset_dict["validation"],
            dataset_dict["test"],
        ])
        adapter.from_dataset_to_dat(merged, args.dst)
        print(f"✅ merged {len(merged)} records")
        print(f"📄 saved to {args.dst}")
    else:
        if adapter_name is None:
            if meta_data is None:
                meta_data = DatasetMetadata.from_directory(args.dst)
            adapter_name = meta_data.adapter

        if adapter_name is None:
            parser.error("--adapter is required or must exist in .hf_meta.json")

        adapter = ADAPTERS[adapter_name]

        dataset_dict = adapter.from_hub_to_dataset_dict(args.src)

        for split_name in ("train", "validation", "test"):
            split_data = dataset_dict[split_name]
            adapter.from_dataset_to_dat(
                dataset=split_data,
                dat_filename=os.path.join(args.dst, f"{split_name}.dat"),
            )
            print(f"✅ {split_name}: {len(split_data)} records")

        api = hub.HfApi()
        info = api.dataset_info(args.src)

        if info.sha is None:
            raise RuntimeError(f"Unable to retrieve commit SHA for dataset '{args.src}'")

        DatasetMetadata.from_values(
            adapter=adapter_name,
            sha=info.sha,
            repo_id=args.src,
        ).save(args.dst)

        for extra in ("README.md", "LICENSE"):
            try:
                # Pylance: Type of hf_hub_download() is partially unknown
                path = hub.hf_hub_download(  # type: ignore[reportUnknownMemberType]
                    args.src, filename=extra, repo_type="dataset", revision=info.sha
                )
            except FileNotFoundError:
                continue
            shutil.copy(path, os.path.join(args.dst, extra))

        print(f"📁 saved to {args.dst}")


def _handle_push(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    Upload a dataset directory using metadata from `.hf_meta.json`.

    Args:
        args (argparse.Namespace):
            Parsed command-line arguments.

        parser (argparse.ArgumentParser):
            The argument parser used for error reporting.
    """
    dir_path = args.dir
    if not os.path.isdir(dir_path):
        parser.error(f"push: directory '{dir_path}' does not exist")

    meta = DatasetMetadata.from_directory(dir_path)

    if meta.repo_id is None or meta.adapter is None:
        parser.error("push: .hf_meta.json must contain 'repo_id' and 'adapter'")

    new_args = argparse.Namespace(
        src=dir_path,
        dst=meta.repo_id,
        adapter=meta.adapter,
        commit_message=args.commit_message,
        seed=42,
        split_ratio=(0.7, 0.15, 0.15),
        y=args.y
    )

    _handle_upload(new_args, parser, meta)


def _handle_pull(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    Download a dataset using metadata from `.hf_meta.json`.

    Args:
        args (argparse.Namespace):
            Parsed command-line arguments.

        parser (argparse.ArgumentParser):
            The argument parser used for error reporting.
    """
    dir_path = args.dir
    if not os.path.isdir(dir_path):
        parser.error(f"push: directory '{dir_path}' does not exist")

    meta = DatasetMetadata.from_directory(dir_path)

    if meta.repo_id is None or meta.adapter is None:
        parser.error("pull: .hf_meta.json must contain 'repo_id' and 'adapter'")

    new_args = argparse.Namespace(
        src=meta.repo_id,
        dst=dir_path,
        adapter=meta.adapter,
        y=True # Overwrite is always implied for pull
    )

    _handle_download(new_args, parser, meta)


def _handle_info(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Display record counts and metadata for a file or directory.

    Args:
        args (argparse.Namespace):
            Parsed command-line arguments.

        parser (argparse.ArgumentParser):
            The argument parser used for error reporting.
    """
    target = args.target
    if os.path.isfile(target):
        if not target.endswith(".dat"):
            parser.error("info: file must end with '.dat'")
        count = _count_dat_records(target)
        print(f"✅ {os.path.basename(target)}: {count} records")
    elif os.path.isdir(target):
        dat_files = [f for f in cast(list[str], os.listdir(target)) if f.endswith(".dat")]
        if not dat_files:
            parser.error("info: directory contains no .dat files")
        for name in sorted(dat_files):
            count = _count_dat_records(os.path.join(target, name))
            print(f"✅ {os.path.splitext(name)[0]}: {count} records")
        meta_data = DatasetMetadata.from_directory(target)
        if meta_data.adapter:
            print(f"🧩 adapter: {meta_data.adapter}")
        if meta_data.last_download:
            print(f"📥 last_download: {meta_data.last_download} (local time)")
        if meta_data.sha:
            print(f"🔐 sha: {meta_data.sha}")
    else:
        parser.error(f"info: '{target}' is not a valid path")

def main() -> None:
    parser = argparse.ArgumentParser(description="fifo-tool-datasets CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # upload
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload a local .dat file or directory to the Hugging Face Hub"
    )
    upload_parser.add_argument("src", help="Local .dat file or directory")
    upload_parser.add_argument("dst", help="Destination dataset on Hugging Face (username/repo)")
    upload_parser.add_argument("--adapter", choices=ADAPTERS.keys())
    upload_parser.add_argument("--commit-message", required=True, help="Commit message for the upload")
    upload_parser.add_argument("--seed", type=int, default=42)
    upload_parser.add_argument("--split-ratio", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"),
                               default=(0.7, 0.15, 0.15),
                               help="Only used when uploading a single .dat file. Ratios must sum to 1.0")
    upload_parser.add_argument("-y", action="store_true", help="Overwrite remote changes")

    # download
    download_parser = subparsers.add_parser(
        "download",
        help="Download a dataset from the Hugging Face Hub"
    )
    download_parser.add_argument("src", help="Source dataset on Hugging Face (username/repo)")
    download_parser.add_argument("dst", help="Destination directory or .dat file")
    download_parser.add_argument("--adapter", choices=ADAPTERS.keys())
    download_parser.add_argument("-y", action="store_true",
                                 help="Overwrite existing local files or directories")

    # push
    push_parser = subparsers.add_parser(
        "push",
        help="Upload a dataset directory using .hf_meta.json",
    )
    push_parser.add_argument("dir", nargs="?", default=".", help="Dataset directory (default: .)")
    push_parser.add_argument("--commit-message", required=True, help="Commit message for the upload")
    push_parser.add_argument("-y", action="store_true", help="Overwrite remote changes")

    # pull
    pull_parser = subparsers.add_parser(
        "pull",
        help="Download a dataset using .hf_meta.json",
    )
    pull_parser.add_argument("dir", nargs="?", default=".", help="Target directory (default: .)")

    # split
    split_parser = subparsers.add_parser(
        "split",
        help="Split a .dat file into train/val/test directory"
    )
    split_parser.add_argument("src", help="Input .dat file")
    split_parser.add_argument("--to", dest="dst", help="Target directory (default: <basename>/)")
    split_parser.add_argument("--adapter", choices=ADAPTERS.keys(), required=True)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--split-ratio", nargs=3, type=float,
                              metavar=('TRAIN', 'VAL', 'TEST'), default=(0.7, 0.15, 0.15),
                              help="Ratios for train/val/test splits. Must sum to 1.0")
    split_parser.add_argument("-y", action="store_true", help="Overwrite existing directory")

    # merge
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge split directory back into a .dat file"
    )
    merge_parser.add_argument("src", help="Input directory with train/val/test .dat files")
    merge_parser.add_argument("--to", dest="dst", help="Output .dat file (default: <dirname>.dat)")
    merge_parser.add_argument("--adapter", choices=ADAPTERS.keys(), required=True)
    merge_parser.add_argument("-y", action="store_true", help="Overwrite existing .dat file")

    # sort
    sort_parser = subparsers.add_parser(
        "sort",
        help="Sort DSL .dat files by system prompt"
    )
    sort_parser.add_argument("path", help=".dat file or directory to sort in place")
    sort_parser.add_argument("--adapter", choices=["dsl"], default="dsl")

    # info
    info_parser = subparsers.add_parser(
        "info",
        help="Show record counts and metadata for a file or directory",
    )
    info_parser.add_argument("target", nargs="?", default=".",
                             help=".dat file or directory (defaults to current directory)")

    args = parser.parse_args()

    if args.command == "upload":
        _handle_upload(args, parser)

    elif args.command == "download":
        _handle_download(args, parser)

    elif args.command == "push":
        _handle_push(args, parser)

    elif args.command == "pull":
        _handle_pull(args, parser)

    elif args.command == "split":
        out_dir = args.dst or os.path.splitext(os.path.basename(args.src))[0]

        if os.path.exists(out_dir):
            existing = cast(list[str], os.listdir(out_dir))
            if existing and not args.y:
                parser.error(f"Directory '{out_dir}' already exists and is not empty."
                             " Use -y to overwrite.")
        os.makedirs(out_dir, exist_ok=True)

        train_ratio, val_ratio, test_ratio = args.split_ratio

        adapter = ADAPTERS[args.adapter]
        try:
            splits = cast(dict[str, Dataset], adapter.from_dat_to_dataset_dict(
                dat_filename=args.src,
                seed=args.seed,
                split_ratios=(train_ratio, val_ratio, test_ratio),
            ))
        except ValueError as e:
            parser.error(str(e))

        total = sum(len(splits[name]) for name in splits.keys())

        for split_name, split_data in splits.items():
            adapter.from_dataset_to_dat(
                dataset=split_data,
                dat_filename=os.path.join(out_dir, f"{split_name}.dat")
            )
            print(f"✅ {split_name}: {len(split_data)} records")

        print(f"📅 total: {total} records")

    elif args.command == "merge":
        if not os.path.isdir(args.src):
            parser.error(
                f"Input path '{args.src}' does not exist or is not a directory. "
                "Expected a directory containing split .dat files."
            )

        adapter = ADAPTERS[args.adapter]
        out_file = args.dst or f"{os.path.basename(os.path.normpath(args.src))}.dat"

        if os.path.exists(out_file) and not args.y:
            parser.error(f"Output file '{out_file}' already exists. Use -y to overwrite.")

        datasets: list[Dataset] = []
        total_records = 0

        for name in ("train", "validation", "test"):
            path = os.path.join(args.src, f"{name}.dat")
            if os.path.exists(path):
                ds = adapter.from_dat_to_dataset(path)
                datasets.append(ds)
                print(f"✅ {name}: {len(ds)} records")
                total_records += len(ds)

        if not datasets:
            parser.error("No valid .dat files found in the directory.")

        merged = concatenate_datasets(datasets)
        adapter.from_dataset_to_dat(merged, out_file)

        print(f"📅 total: {total_records} records")

    elif args.command == "sort":
        adapter = ADAPTERS[args.adapter]
        if not isinstance(adapter, DSLAdapter):
            parser.error("The sort command currently supports only the 'dsl' adapter.")

        target = args.path
        if os.path.isdir(target):
            files = [f for f in Path(target).iterdir() if f.suffix == ".dat"]
            if not files:
                parser.error("No .dat files found in the directory.")
            for file in files:
                adapter.sort_dat_file(str(file))
                print(f"✅ sorted {file}")
        elif os.path.isfile(target):
            adapter.sort_dat_file(target)
            print(f"✅ sorted {target}")
        else:
            parser.error(f"Path '{target}' does not exist")

    elif args.command == "info":
        _handle_info(args, parser)

if __name__ == "__main__":
    main()
