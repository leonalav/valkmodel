from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data.dataset_registry import DatasetRegistry
from data.streaming_dataset import iter_tokenized_documents, load_stream_rows
from data.tokenizer_setup import load_tokenizer

from .arrow_writer import ArrowDocWriter
from .parquet_packer import ParquetPacker
from .registry_filter import build_pretok_registry

DEFAULT_HF_REPO_ID = "leonidas123/valkmodel-data"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="valkmodel-pretok")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--output-dir", required=True)
    build.add_argument("--tokenizer", required=True)
    build.add_argument("--dataset", action="append", required=True)
    build.add_argument("--stage", action="append", type=int, required=True)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--output-dir", required=True)
    publish.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    publish.add_argument("--revision")

    return parser


def iter_tokenized_documents_for_spec(spec: Any, tokenizer: Any, limit: int | None = None):
    rows = load_stream_rows(spec)
    for index, tokens in enumerate(iter_tokenized_documents(rows, spec, tokenizer)):
        if limit is not None and index >= limit:
            break
        yield tokens


def _tokenizer_for_name(tokenizer_name_or_path: str) -> Any:
    return load_tokenizer(tokenizer_name_or_path)


def _build_datasets(registry: DatasetRegistry, dataset_names: list[str]) -> list[str]:
    names = []
    for name in dataset_names:
        registry.get(name)
        names.append(name)
    return names


def _build_pretok_output(output_dir: Path, tokenizer_name: str, dataset_names: list[str], stages: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "canonical").mkdir(parents=True, exist_ok=True)
    (output_dir / "packed").mkdir(parents=True, exist_ok=True)

    registry = build_pretok_registry(DatasetRegistry())
    tokenizer = _tokenizer_for_name(tokenizer_name)
    datasets = _build_datasets(registry, dataset_names)

    for dataset_name in datasets:
        spec = registry.get(dataset_name)
        docs = list(iter_tokenized_documents_for_spec(spec, tokenizer))

        with (output_dir / "canonical" / f"{dataset_name}.arrow").open("wb") as sink:
            with ArrowDocWriter(sink) as writer:
                for doc in docs:
                    writer.add_document(doc)

        for stage in stages:
            stage_dir = output_dir / "packed" / f"stage_{stage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            with (stage_dir / f"{dataset_name}.parquet").open("wb") as sink:
                with ParquetPacker(sink, block_size=stage, eos_token_id=tokenizer.eos_token_id) as packer:
                    for doc in docs:
                        packer.add_document(doc)

    (output_dir / "manifest.json").write_text(json.dumps({"datasets": datasets, "stages": stages}, indent=2), encoding="utf-8")


def upload_output_to_hf(output_dir: str | Path, repo_id: str, revision: str | None = None) -> None:
    try:
        from huggingface_hub import upload_folder
    except Exception as exc:  # pragma: no cover
        raise ImportError("huggingface_hub is required to publish pretokenized artifacts") from exc
    upload_folder(repo_id=repo_id, folder_path=str(output_dir), revision=revision)


def _publish_output(output_dir: str | Path, repo_id: str, revision: str | None = None) -> None:
    upload_output_to_hf(output_dir, repo_id, revision=revision)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        _build_pretok_output(Path(args.output_dir), args.tokenizer, args.dataset, args.stage)
        return 0
    if args.command == "publish":
        _publish_output(args.output_dir, args.repo_id, revision=args.revision)
        return 0
    raise ValueError(f"unknown command: {args.command}")
