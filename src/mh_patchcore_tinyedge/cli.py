"""Command line interface."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from mh_patchcore_tinyedge.bank import build_student_bank, load_memory_bank, save_memory_bank
from mh_patchcore_tinyedge.checkpoint import load_checkpoint
from mh_patchcore_tinyedge.devices import resolve_device
from mh_patchcore_tinyedge.heatmap import save_heatmap
from mh_patchcore_tinyedge.images import collect_image_paths
from mh_patchcore_tinyedge.scoring import score_images


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    console = Console(highlight=False)
    try:
        if args.command == "bank":
            return _bank_command(args, console)
        if args.command == "score":
            return _score_command(args, console)
        if args.command == "inspect":
            return _inspect_command(args, console)
    except Exception as exc:
        console.print(f"[red]error:[/] {exc}")
        return 1
    parser.error("missing command")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mh-tinyedge")
    subparsers = parser.add_subparsers(dest="command")

    bank_parser = subparsers.add_parser("bank", help="build or inspect a memory bank")
    bank_subparsers = bank_parser.add_subparsers(dest="bank_command", required=True)
    build_parser = bank_subparsers.add_parser("build", help="build a student memory bank")
    build_parser.add_argument("--checkpoint", required=True, type=Path)
    build_parser.add_argument("--images", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument("--device", default="auto")
    build_parser.add_argument("--batch-size", default=8, type=int)

    score_parser = subparsers.add_parser("score", help="score one image or a folder")
    score_parser.add_argument("--checkpoint", required=True, type=Path)
    score_parser.add_argument("--bank", required=True, type=Path)
    score_parser.add_argument("--input", required=True, type=Path)
    score_parser.add_argument("--output", required=True, type=Path)
    score_parser.add_argument("--heatmap-dir", type=Path)
    score_parser.add_argument("--device", default="auto")
    score_parser.add_argument("--prototype-chunk-size", default=4096, type=int)

    inspect_parser = subparsers.add_parser("inspect", help="show checkpoint and bank metadata")
    inspect_parser.add_argument("--checkpoint", required=True, type=Path)
    inspect_parser.add_argument("--bank", required=True, type=Path)
    inspect_parser.add_argument("--device", default="cpu")
    return parser


def _bank_command(args: argparse.Namespace, console: Console) -> int:
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    images = collect_image_paths(args.images)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    progress = _progress(console)
    with progress:
        task = progress.add_task("building memory bank", total=len(images))
        bank = build_student_bank(
            checkpoint=checkpoint,
            image_paths=images,
            device=device,
            batch_size=args.batch_size,
            progress=progress,
            task_id=task,
        )
    save_memory_bank(args.output, bank)
    table = Table(title="Memory Bank")
    table.add_column("field")
    table.add_column("value")
    table.add_row("images", str(len(images)))
    table.add_row("prototypes", str(bank.features.shape[0]))
    table.add_row("embedding_dim", str(bank.embedding_dim))
    table.add_row("grid", f"{bank.output_grid_hw[0]}x{bank.output_grid_hw[1]}")
    table.add_row("device", str(device))
    table.add_row("output", str(args.output))
    console.print(table)
    return 0


def _score_command(args: argparse.Namespace, console: Console) -> int:
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    bank = load_memory_bank(args.bank)
    images = collect_image_paths(args.input)
    progress = _progress(console)
    with progress:
        task = progress.add_task("scoring images", total=len(images))
        scores = score_images(
            checkpoint=checkpoint,
            bank=bank,
            image_paths=images,
            device=device,
            prototype_chunk_size=args.prototype_chunk_size,
            progress=progress,
            task_id=task,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "score", "heatmap"])
        writer.writeheader()
        for index, result in enumerate(scores):
            heatmap_path = ""
            if args.heatmap_dir is not None:
                heatmap_file = args.heatmap_dir / f"{index:06d}_{result.path.stem}.png"
                save_heatmap(heatmap_file, result.patch_scores)
                heatmap_path = str(heatmap_file)
            writer.writerow(
                {
                    "path": str(result.path),
                    "score": f"{result.score:.10g}",
                    "heatmap": heatmap_path,
                }
            )
    _print_score_summary(console, scores, args.output)
    return 0


def _inspect_command(args: argparse.Namespace, console: Console) -> int:
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    bank = load_memory_bank(args.bank)
    table = Table(title="TinyEdge Runtime")
    table.add_column("field")
    table.add_column("value")
    table.add_row("checkpoint", str(checkpoint.path))
    table.add_row("bank_prototypes", str(bank.features.shape[0]))
    table.add_row("checkpoint_embedding_dim", str(checkpoint.embedding_dim))
    table.add_row("bank_embedding_dim", str(bank.embedding_dim))
    table.add_row("checkpoint_grid", f"{checkpoint.output_grid_hw[0]}x{checkpoint.output_grid_hw[1]}")
    table.add_row("bank_grid", f"{bank.output_grid_hw[0]}x{bank.output_grid_hw[1]}")
    table.add_row("device", str(device))
    console.print(table)
    return 0


def _print_score_summary(console: Console, scores: Sequence[object], output: Path) -> None:
    values = [float(item.score) for item in scores]
    table = Table(title="Scores")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("images", str(len(values)))
    table.add_row("min", f"{min(values):.6g}")
    table.add_row("max", f"{max(values):.6g}")
    table.add_row("mean", f"{sum(values) / len(values):.6g}")
    table.add_row("output", str(output))
    console.print(table)


def _progress(console: Console) -> Progress:
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


if __name__ == "__main__":
    raise SystemExit(main())
