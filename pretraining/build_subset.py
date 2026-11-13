from __future__ import annotations

import argparse
import io
import math
import tarfile
from pathlib import Path
from typing import Iterable


def load_allowed_ids(file_path: str) -> set[str]:
    """
    Load exact sample IDs from a text file.

    Args:
        file_path: Path to a text file containing one exact sample ID per line.

    Returns:
        A set of allowed sample IDs.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Allowed ID file not found: {path}")

    allowed_ids = {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    }

    if not allowed_ids:
        raise ValueError(f"No IDs found in {path}")

    return allowed_ids


def shard_paths(root: Path, split: str, modality: str) -> list[Path]:
    """
    List shard paths for one modality.

    Args:
        root: Root directory of SSL4EO-S12-v1.1.
        split: Dataset split, e.g. 'train'.
        modality: Modality name, e.g. 'S2L2A' or 'S1GRD'.

    Returns:
        Sorted list of shard tar paths.
    """
    modality_dir = root / split / modality
    if not modality_dir.exists():
        raise FileNotFoundError(f"Directory not found: {modality_dir}")

    shards = sorted(modality_dir.glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No shard files found in {modality_dir}")

    return shards


def member_sample_id(member_name: str) -> str:
    """
    Extract sample ID from a tar member name.

    Example:
        ssl4eos12_train_seasonal_data_0000006.zarr.zip
        -> ssl4eos12_train_seasonal_data_0000006

    Args:
        member_name: Tar member name.

    Returns:
        Sample ID without extension.
    """
    name = Path(member_name).name

    if name.endswith(".zarr.zip"):
        return name[:-9]

    return Path(name).stem


def iter_matching_members(
    tar_path: Path,
    allowed_ids: set[str],
) -> Iterable[tuple[tarfile.TarInfo, bytes]]:
    """
    Iterate over matching tar members and return their metadata plus raw bytes.

    Args:
        tar_path: Input tar file.
        allowed_ids: Set of exact sample IDs to keep.

    Yields:
        Tuples of (TarInfo, raw_bytes) for matching members.
    """
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue

            sample_id = member_sample_id(member.name)
            if sample_id not in allowed_ids:
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            payload = extracted.read()

            new_info = tarfile.TarInfo(name=member.name)
            new_info.size = len(payload)
            new_info.mtime = member.mtime
            new_info.mode = member.mode

            yield new_info, payload


def write_subset_shards(
    input_root: Path,
    output_root: Path,
    split: str,
    allowed_ids: set[str],
    modalities: list[str],
    max_samples_per_output_shard: int = 1000,
) -> None:
    """
    Build a filtered subset dataset by copying only selected samples.

    The script keeps modality alignment by scanning shard pairs with the same
    index for all requested modalities.

    Args:
        input_root: Root directory of the original dataset.
        output_root: Root directory of the subset dataset.
        split: Dataset split, e.g. 'train'.
        allowed_ids: Set of exact sample IDs to keep.
        modalities: Modalities to copy.
        max_samples_per_output_shard: Maximum number of samples per output tar.
    """
    modality_to_shards = {
        modality: shard_paths(input_root, split, modality)
        for modality in modalities
    }

    n_shards = len(next(iter(modality_to_shards.values())))
    for modality, paths in modality_to_shards.items():
        if len(paths) != n_shards:
            raise ValueError(
                f"Modality {modality} has {len(paths)} shards, expected {n_shards}."
            )

    for modality in modalities:
        (output_root / split / modality).mkdir(parents=True, exist_ok=True)

    kept_total = 0
    current_out_index = 1
    current_count = 0
    out_tars: dict[str, tarfile.TarFile] = {}

    def open_output_tars(index: int) -> dict[str, tarfile.TarFile]:
        opened = {}
        for modality in modalities:
            out_path = (
                output_root
                / split
                / modality
                / f"ssl4eos12_train_{index:06d}.tar"
            )
            opened[modality] = tarfile.open(out_path, "w")
        return opened

    def close_output_tars(tars: dict[str, tarfile.TarFile]) -> None:
        for tar in tars.values():
            tar.close()

    out_tars = open_output_tars(current_out_index)

    try:
        for shard_idx in range(n_shards):
            per_modality_matches: dict[str, list[tuple[tarfile.TarInfo, bytes]]] = {}

            for modality in modalities:
                tar_path = modality_to_shards[modality][shard_idx]
                matches = list(iter_matching_members(tar_path, allowed_ids))
                per_modality_matches[modality] = matches

            sample_names_per_modality = {
                modality: [member_sample_id(info.name) for info, _ in matches]
                for modality, matches in per_modality_matches.items()
            }

            reference_modality = modalities[0]
            reference_ids = sample_names_per_modality[reference_modality]

            for modality in modalities[1:]:
                if sample_names_per_modality[modality] != reference_ids:
                    raise ValueError(
                        f"Shard alignment mismatch in shard index {shard_idx} "
                        f"between {reference_modality} and {modality}."
                    )

            for sample_pos, sample_id in enumerate(reference_ids):
                if current_count >= max_samples_per_output_shard:
                    close_output_tars(out_tars)
                    current_out_index += 1
                    current_count = 0
                    out_tars = open_output_tars(current_out_index)

                for modality in modalities:
                    info, payload = per_modality_matches[modality][sample_pos]
                    out_tars[modality].addfile(info, io.BytesIO(payload))

                current_count += 1
                kept_total += 1

            print(
                f"Processed input shard {shard_idx + 1}/{n_shards} | "
                f"kept so far: {kept_total}",
                flush=True,
            )

    finally:
        close_output_tars(out_tars)

    print(f"Finished. Total kept samples: {kept_total}", flush=True)
    print(f"Subset written to: {output_root}", flush=True)


def main() -> None:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--allowed_ids", type=str, required=True)
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["S2L2A", "S1GRD"],
    )
    parser.add_argument(
        "--max_samples_per_output_shard",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    allowed_ids = load_allowed_ids(args.allowed_ids)

    write_subset_shards(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        split=args.split,
        allowed_ids=allowed_ids,
        modalities=args.modalities,
        max_samples_per_output_shard=args.max_samples_per_output_shard,
    )


if __name__ == "__main__":
    main()
