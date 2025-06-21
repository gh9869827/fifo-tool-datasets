import argparse
import os
from pathlib import Path
from typing import cast
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

def main() -> None:
    parser = argparse.ArgumentParser(description="fifo-tool-datasets CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # copy
    copy_parser = subparsers.add_parser("copy", help="Upload/download between .dat and Hugging Face")
    copy_parser.add_argument("src", help="Source .dat file, directory or Hugging Face dataset name")
    copy_parser.add_argument("dst", help="Destination Hugging Face dataset name, directory or .dat file")
    copy_parser.add_argument("--adapter", choices=ADAPTERS.keys(), required=True)
    copy_parser.add_argument("--commit-message", help="Required if uploading to Hugging Face")
    copy_parser.add_argument("--seed", type=int, default=None)
    copy_parser.add_argument("--split-ratio", nargs=3, type=float, metavar=('TRAIN', 'VAL', 'TEST'),
                             default=(0.7, 0.15, 0.15),
                             help="Only used when uploading a single .dat file. Ratios must sum to 1.0")
    copy_parser.add_argument("-y", action="store_true", help="Overwrite existing local files or directories")

    # split
    split_parser = subparsers.add_parser("split", help="Split a .dat file into train/val/test directory")
    split_parser.add_argument("src", help="Input .dat file")
    split_parser.add_argument("--to", dest="dst", help="Target directory (default: <basename>/)")
    split_parser.add_argument("--adapter", choices=ADAPTERS.keys(), required=True)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--split-ratio", nargs=3, type=float, metavar=('TRAIN', 'VAL', 'TEST'),
                              default=(0.7, 0.15, 0.15), help="Ratios for train/val/test splits. Must sum to 1.0")
    split_parser.add_argument("-y", action="store_true", help="Overwrite existing directory")

    # merge
    merge_parser = subparsers.add_parser("merge", help="Merge split directory back into a .dat file")
    merge_parser.add_argument("src", help="Input directory with train/val/test .dat files")
    merge_parser.add_argument("--to", dest="dst", help="Output .dat file (default: <dirname>.dat)")
    merge_parser.add_argument("--adapter", choices=ADAPTERS.keys(), required=True)
    merge_parser.add_argument("-y", action="store_true", help="Overwrite existing .dat file")

    # sort
    sort_parser = subparsers.add_parser("sort", help="Sort DSL .dat files by system prompt")
    sort_parser.add_argument("path", help=".dat file or directory to sort in place")
    sort_parser.add_argument("--adapter", choices=["dsl"], default="dsl")

    args = parser.parse_args()
    adapter = ADAPTERS[args.adapter]

    if args.command == "copy":
        src_is_file = os.path.isfile(args.src) and args.src.endswith(".dat")
        src_is_dir = os.path.isdir(args.src)
        dst_is_file = args.dst.endswith(".dat")
        dst_is_dir = not dst_is_file

        if src_is_file and dst_is_dir:
            # file → hub (split + upload)
            if not args.commit_message:
                parser.error("--commit-message is required when uploading to Hugging Face")
            adapter.from_dat_to_hub(
                dat_filename=args.src,
                hub_dataset=args.dst,
                commit_message=args.commit_message,
                seed=args.seed,
                split_ratios=tuple(args.split_ratio)  # already parsed to float[3]
            )

        elif src_is_dir and dst_is_dir:
            # dir → hub (upload as-is)
            if not args.commit_message:
                parser.error("--commit-message is required when uploading to Hugging Face")
            adapter.from_dir_to_hub(
                dat_dir=args.src,
                hub_dataset=args.dst,
                commit_message=args.commit_message
            )

        elif not src_is_file and dst_is_file:
            # hub → single .dat file (download + merge)
            if os.path.exists(args.dst) and not args.y:
                parser.error(
                    f"Output file '{args.dst}' already exists. Use -y to overwrite."
                )

            dataset_dict = adapter.from_hub_to_dataset_dict(args.src)

            merged = concatenate_datasets([
                dataset_dict["train"],
                dataset_dict["validation"],
                dataset_dict["test"]
            ])

            adapter.from_dataset_to_dat(merged, args.dst)
            print(f"✅ merged {len(merged)} records")
            print(f"📄 saved to {args.dst}")

        elif not src_is_file and dst_is_dir:
            # hub → dir (download splits and write to disk)
            if os.path.exists(args.dst):
                existing = cast(list[str], os.listdir(args.dst))
                if existing and not args.y:
                    parser.error(
                        f"Directory '{args.dst}' already exists and is not empty."
                        " Use -y to overwrite."
                    )
            os.makedirs(args.dst, exist_ok=True)

            dataset_dict = adapter.from_hub_to_dataset_dict(args.src)

            for split_name in ("train", "validation", "test"):
                split_data = dataset_dict[split_name]
                adapter.from_dataset_to_dat(
                    dataset=split_data,
                    dat_filename=os.path.join(args.dst, f"{split_name}.dat")
                )
                print(f"✅ {split_name}: {len(split_data)} records")

            print(f"📁 saved to {args.dst}")

        else:
            parser.error("Unsupported source/destination combination.")

    elif args.command == "split":
        out_dir = args.dst or os.path.splitext(os.path.basename(args.src))[0]

        if os.path.exists(out_dir):
            existing = cast(list[str], os.listdir(out_dir))
            if existing and not args.y:
                parser.error(f"Directory '{out_dir}' already exists and is not empty."
                             " Use -y to overwrite.")
        os.makedirs(out_dir, exist_ok=True)

        train_ratio, val_ratio, test_ratio = args.split_ratio

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

if __name__ == "__main__":
    main()
