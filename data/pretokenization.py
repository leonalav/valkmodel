from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
from collections.abc import Iterator
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.ipc as pa_ipc
from datasets import Dataset, IterableDataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

try:
    from data.dataset_registry import DEFAULT_DATASET_SPECS, DatasetSpec
    from data.streaming_dataset import build_dataset_load_kwargs, extract_text
except ModuleNotFoundError:
    from dataset_registry import DEFAULT_DATASET_SPECS, DatasetSpec
    from streaming_dataset import build_dataset_load_kwargs, extract_text

EXCLUDED_DATASETS = {"scientific_long", "scientific_papers", "creative_writing"}
DEFAULT_CURRICULUM_STAGES = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
DEFAULT_TOKENIZER = "meta-llama/Meta-Llama-3-8B"
DEFAULT_OUTPUT_DIR = Path("data/pretokenized")
DEFAULT_CACHE_DIR = Path(".cache/hf_datasets_pretok")
DEFAULT_HUB_REPO_ID = "leonidas123/valkmodel-data"
DEFAULT_HUB_REPO_TYPE = "dataset"
DEFAULT_SHARD_SIZE = 10_000
MIN_CHARS = 64
TEXT_COLUMN = "text"
_WORKER_TOKENIZER: PreTrainedTokenizerBase | None = None


@dataclass(frozen=True)
class PretokenizationSpec:
    name: str
    hf_path: str
    split: str = "train"
    text_field: str = "text"
    language: str | None = None
    subset: str | None = None
    revision: str | None = None
    output_format: str = "parquet"

    @classmethod
    def from_registry_spec(cls, spec: DatasetSpec) -> "PretokenizationSpec":
        return cls(
            name=spec.name,
            hf_path=spec.hf_path,
            split=spec.split,
            text_field=spec.text_field,
            language=spec.language,
            subset=spec.subset,
            revision=spec.revision,
            output_format="arrow" if spec.hf_path == "bigcode/the-stack-v2" else "parquet",
        )

    def load_kwargs(self, cache_dir: Path, streaming: bool) -> dict[str, Any]:
        registry_spec = DatasetSpec(
            name=self.name,
            hf_path=self.hf_path,
            split=self.split,
            text_field=self.text_field,
            language=self.language,
            subset=self.subset,
            revision=self.revision,
        )
        kwargs = build_dataset_load_kwargs(registry_spec)
        kwargs["streaming"] = streaming
        kwargs["cache_dir"] = str(cache_dir)
        return kwargs


def included_specs() -> list[PretokenizationSpec]:
    specs = []
    for name, spec in DEFAULT_DATASET_SPECS.items():
        if name in EXCLUDED_DATASETS or spec.hf_path in EXCLUDED_DATASETS:
            continue
        specs.append(PretokenizationSpec.from_registry_spec(spec))
    return specs


def terminal_supports_unicode() -> bool:
    encoding = getattr(sys.stderr, "encoding", None) or getattr(sys.stdout, "encoding", None) or ""
    return "UTF" in encoding.upper()


def progress_bar(iterable: Iterable[Any], *, total: int | None, desc: str):
    ascii_bar = not terminal_supports_unicode()
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit="docs",
        dynamic_ncols=True,
        leave=True,
        ascii=ascii_bar,
        mininterval=0.5,
        smoothing=0.05,
    )


def output_root_for_spec(root: Path, spec: PretokenizationSpec) -> Path:
    subset = spec.subset or spec.language or "default"
    return root / spec.name / subset


def success_marker(output_dir: Path) -> Path:
    return output_dir / "_SUCCESS"


def is_complete(output_dir: Path) -> bool:
    return success_marker(output_dir).exists()


def prepare_text(row: dict[str, Any], spec: PretokenizationSpec) -> dict[str, str | None]:
    registry_spec = DatasetSpec(
        name=spec.name,
        hf_path=spec.hf_path,
        split=spec.split,
        text_field=spec.text_field,
        language=spec.language,
        subset=spec.subset,
        revision=spec.revision,
    )
    text = extract_text(row, registry_spec)
    if text is None:
        return {TEXT_COLUMN: None}
    text = text.strip()
    if len(text) < MIN_CHARS:
        return {TEXT_COLUMN: None}
    return {TEXT_COLUMN: text}


def has_text(row: dict[str, Any]) -> bool:
    return row[TEXT_COLUMN] is not None


def init_tokenizer_worker(tokenizer_name_or_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, trust_remote_code=False)


def tokenize_texts(texts: list[str], tokenizer: PreTrainedTokenizerBase) -> list[list[int]]:
    clean_texts = [text for text in texts if isinstance(text, str) and text.strip()]
    if not clean_texts:
        return []
    encoded = tokenizer(clean_texts, add_special_tokens=False, return_attention_mask=False)
    return [list(tokens) for tokens in encoded["input_ids"] if tokens]


def tokenize_worker(texts: list[str]) -> list[list[int]]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return tokenize_texts(texts, _WORKER_TOKENIZER)


def text_batches(dataset: Dataset, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        yield dataset[start : min(start + batch_size, len(dataset))][TEXT_COLUMN]


def iter_tokenized_batches(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_name_or_path: str,
    spec_name: str,
    batch_size: int,
    num_proc: int,
):
    total = (len(dataset) + batch_size - 1) // batch_size
    batches = text_batches(dataset, batch_size)
    if num_proc <= 1:
        iterator = (tokenize_texts(batch, tokenizer) for batch in batches)
        yield from progress_bar(iterator, total=total, desc=f"{spec_name}: tokenizing")
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=num_proc, initializer=init_tokenizer_worker, initargs=(tokenizer_name_or_path,)) as pool:
            iterator = pool.imap(tokenize_worker, batches)
            yield from progress_bar(iterator, total=total, desc=f"{spec_name}: tokenizing")


def write_shard(rows: dict[str, list[Any]], output_dir: Path, fmt: str, shard_id: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = "arrow" if fmt == "arrow" else "parquet"
    shard_path = output_dir / f"shard-{shard_id:06d}.{extension}"
    shard = Dataset.from_dict(rows)
    if fmt == "arrow":
        table = pa.Table.from_pydict({column: shard[column] for column in shard.column_names})
        with pa_ipc.new_file(str(shard_path), table.schema) as writer:
            writer.write_table(table)
    else:
        shard.to_parquet(str(shard_path))


def flush_full_shards(rows: dict[str, list[Any]], output_dir: Path, fmt: str, shard_size: int, shard_id: int) -> int:
    while len(rows["input_ids"]) >= shard_size:
        shard_rows = {key: value[:shard_size] for key, value in rows.items()}
        write_shard(shard_rows, output_dir, fmt, shard_id)
        shard_id += 1
        for key in rows:
            rows[key] = rows[key][shard_size:]
    return shard_id


def iter_packed_blocks(token_batches, stage: int, eos_token_id: int) -> Iterator[list[int]]:
    buffer: list[int] = []
    for tokens_batch in token_batches:
        for tokens in tokens_batch:
            buffer.extend(tokens)
            buffer.append(eos_token_id)
            while len(buffer) >= stage:
                yield buffer[:stage]
                del buffer[:stage]


def write_curriculum_stage(
    token_batches,
    output_dir: Path,
    fmt: str,
    stage: int,
    eos_token_id: int,
    shard_size: int,
) -> None:
    rows: dict[str, list[Any]] = {"input_ids": [], "token_count": [], "seq_len": []}
    shard_id = 0
    for block in iter_packed_blocks(token_batches, stage, eos_token_id):
        rows["input_ids"].append(block)
        rows["token_count"].append(stage)
        rows["seq_len"].append(stage)
        shard_id = flush_full_shards(rows, output_dir, fmt, shard_size, shard_id)
    if rows["input_ids"]:
        write_shard(rows, output_dir, fmt, shard_id)


def hub_path_exists(api: HfApi, repo_id: str, repo_type: str, path_in_repo: str) -> bool:
    try:
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=f"{path_in_repo}/_SUCCESS",
        )
        return True
    except (EntryNotFoundError, RepositoryNotFoundError):
        return False


def upload_directory(api: HfApi, local_dir: Path, repo_id: str, repo_type: str, path_in_repo: str) -> None:
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=f"Upload {path_in_repo}",
    )


def remove_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def remove_dataset_cache(dataset: Dataset | IterableDataset | None, cache_dir: Path) -> None:
    if dataset is not None and hasattr(dataset, "cleanup_cache_files"):
        try:
            dataset.cleanup_cache_files()
        except (OSError, RuntimeError):
            pass
    gc.collect()
    remove_path(cache_dir)


def load_raw_dataset(spec: PretokenizationSpec, cache_dir: Path, streaming: bool) -> Dataset | IterableDataset:
    return load_dataset(**spec.load_kwargs(cache_dir=cache_dir, streaming=streaming))


def materialize_iterable(dataset: IterableDataset, limit: int | None) -> Dataset:
    rows = []
    iterator = iter(dataset.take(limit)) if limit is not None else iter(dataset)
    for row in iterator:
        rows.append(row)
    return Dataset.from_list(rows)


def tokenize_spec(
    spec: PretokenizationSpec,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_name_or_path: str,
    output_root: Path,
    cache_root: Path,
    stages: tuple[int, ...],
    num_proc: int,
    batch_size: int,
    shard_size: int,
    streaming: bool,
    limit: int | None,
    hub_api: HfApi,
    hub_repo_id: str,
    hub_repo_type: str,
) -> None:
    spec_output_dir = output_root_for_spec(output_root, spec)
    path_in_repo = spec_output_dir.relative_to(output_root).as_posix()
    if is_complete(spec_output_dir):
        print(f"Skipping {spec.name}: found {success_marker(spec_output_dir)}")
        upload_directory(hub_api, spec_output_dir, hub_repo_id, hub_repo_type, path_in_repo)
        remove_path(spec_output_dir)
        return
    if hub_path_exists(hub_api, hub_repo_id, hub_repo_type, path_in_repo):
        print(f"Skipping {spec.name}: found {path_in_repo}/_SUCCESS in {hub_repo_id}")
        remove_path(spec_output_dir)
        return

    cache_dir = cache_root / spec.name
    raw_dataset: Dataset | IterableDataset | None = None
    prepared: Dataset | None = None
    try:
        raw_dataset = load_raw_dataset(spec, cache_dir=cache_dir, streaming=streaming)
        if isinstance(raw_dataset, IterableDataset):
            raw_dataset = materialize_iterable(raw_dataset, limit)
        elif limit is not None:
            raw_dataset = raw_dataset.select(range(min(limit, len(raw_dataset))))

        prepare_proc = num_proc if num_proc > 1 else None
        prepared = raw_dataset.map(
            prepare_text,
            fn_kwargs={"spec": spec},
            remove_columns=list(raw_dataset.column_names),
            num_proc=prepare_proc,
            desc=f"{spec.name}: extracting text",
        )
        prepared = prepared.filter(has_text, num_proc=prepare_proc)

        pending_stage_dirs = {}
        for stage in stages:
            stage_dir = spec_output_dir / f"seq_{stage}"
            stage_path_in_repo = f"{path_in_repo}/seq_{stage}"
            if is_complete(stage_dir):
                print(f"Skipping {spec.name} seq_{stage}: found {success_marker(stage_dir)}")
            elif hub_path_exists(hub_api, hub_repo_id, hub_repo_type, stage_path_in_repo):
                print(f"Skipping {spec.name} seq_{stage}: found {stage_path_in_repo}/_SUCCESS in {hub_repo_id}")
            else:
                pending_stage_dirs[stage] = stage_dir

        for stage, stage_dir in pending_stage_dirs.items():
            token_batches = iter_tokenized_batches(prepared, tokenizer, tokenizer_name_or_path, spec.name, batch_size, num_proc)
            write_curriculum_stage(token_batches, stage_dir, spec.output_format, stage, tokenizer.eos_token_id, shard_size)
            success_marker(stage_dir).write_text("ok\n", encoding="utf-8")
            gc.collect()

        spec_output_dir.mkdir(parents=True, exist_ok=True)
        success_marker(spec_output_dir).write_text("ok\n", encoding="utf-8")
        upload_directory(hub_api, spec_output_dir, hub_repo_id, hub_repo_type, path_in_repo)
        remove_path(spec_output_dir)
    finally:
        remove_dataset_cache(prepared, cache_dir)
        remove_dataset_cache(raw_dataset, cache_dir)


def parse_stages(value: str) -> tuple[int, ...]:
    stages = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not stages:
        raise argparse.ArgumentTypeError("at least one curriculum stage is required")
    if any(stage <= 0 for stage in stages):
        raise argparse.ArgumentTypeError("curriculum stages must be positive integers")
    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretokenize Valkmodel datasets for curriculum training.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--stages", type=parse_stages, default=DEFAULT_CURRICULUM_STAGES)
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset names to tokenize. Defaults to every included dataset.")
    parser.add_argument("--num-proc", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--hub-repo-id", default=DEFAULT_HUB_REPO_ID)
    parser.add_argument("--hub-repo-type", default=DEFAULT_HUB_REPO_TYPE)
    parser.add_argument("--streaming", action="store_true", help="Stream then materialize rows before multiprocessing tokenization.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-dataset row limit for dry runs or probes.")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    specs = included_specs()
    if args.datasets:
        requested = set(args.datasets)
        known = {spec.name for spec in specs}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown or excluded datasets: {', '.join(unknown)}")
        specs = [spec for spec in specs if spec.name in requested]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for document boundaries.")

    hub_api = HfApi()
    hub_api.create_repo(repo_id=args.hub_repo_id, repo_type=args.hub_repo_type, exist_ok=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        tokenize_spec(
            spec=spec,
            tokenizer=tokenizer,
            tokenizer_name_or_path=args.tokenizer,
            output_root=args.output_dir,
            cache_root=args.cache_dir,
            stages=args.stages,
            num_proc=args.num_proc,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            streaming=args.streaming,
            limit=args.limit,
            hub_api=hub_api,
            hub_repo_id=args.hub_repo_id,
            hub_repo_type=args.hub_repo_type,
        )


if __name__ == "__main__":
    main()
