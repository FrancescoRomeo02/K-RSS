"""CLI embedding pipeline for scraped videos.

Usage:
    python -m source.AI_RM.embed_pipeline --input /path/to/scraped_videos.json --batch-size 64

The script loads the scraped JSON (either a dict with key "videos" or a list),
initializes the `Embedder` and `EmbeddingStore` from this package, and
processes videos in batches. On `--dry-run` it only prints a summary and examples.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from source.AI_RM.config import default_config
from source.AI_RM.embedder import Embedder
from source.AI_RM.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


def chunked(iterable: Iterable, size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_videos(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        videos = data.get("videos") or data.get("Videos") or []
        if not isinstance(videos, list):
            raise ValueError("Expected 'videos' to be a list in the input JSON")
        return videos

    if isinstance(data, list):
        return data

    raise ValueError("Input JSON must be either a list or a dict with a 'videos' list")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Embed scraped videos and store vectors")
    default_input = Path(default_config.data_path) / "raw" / "scraped_videos.json"
    parser.add_argument("--input", type=Path, default=default_input, help="Path to scraped_videos.json")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to the store, only show summary")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of videos per embedding batch")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not skip videos already present in the store")
    parser.add_argument("--update", action="store_true", help="Recompute embeddings for videos already present (delete+reinsert)")
    parser.set_defaults(resume=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        videos = load_videos(args.input)
    except Exception as e:
        logger.exception("Failed to load videos: %s", e)
        return 2

    logger.info("Loaded %d videos from %s", len(videos), args.input)

    try:
        embedder = Embedder(config=default_config.embedding)
    except Exception as e:
        logger.exception("Failed to initialize Embedder: %s", e)
        return 3

    try:
        store = EmbeddingStore(config=default_config.vector_store)
    except Exception as e:
        logger.exception("Failed to initialize EmbeddingStore: %s", e)
        return 4

    total_scanned = 0
    total_added = 0
    skipped_existing = 0
    skipped_no_id = 0
    sample_added_ids: List[str] = []

    # Filter and deduplicate videos to process
    to_process: List[Dict[str, Any]] = []
    for v in videos:
        total_scanned += 1
        if not isinstance(v, dict):
            logger.debug("Skipping non-dict video entry at index %d", total_scanned - 1)
            skipped_no_id += 1
            continue
        video_id = v.get("video_id")
        if not video_id:
            logger.debug("Skipping video without 'video_id': %s", v.get("title", "(no title)"))
            skipped_no_id += 1
            continue
        try:
            exists = False
            if hasattr(store, "exists"):
                exists = store.exists(video_id)
            elif hasattr(store, "contains"):  # fallback name
                exists = store.get_video(video_id)
            else:
                exists = False
        except Exception:
            logger.exception("Error checking existence for %s; will attempt to add", video_id)
            exists = False

        if exists:
            if args.update:
                # Include existing items for update
                to_process.append(v)
            else:
                skipped_existing += 1
                continue
        else:
            to_process.append(v)

    logger.info("To process: %d videos (skipped existing: %d, no id: %d)", len(to_process), skipped_existing, skipped_no_id)

    if args.dry_run:
        # In dry-run mode we just report and sample
        sample_ids = [v.get("video_id") for v in to_process[:5]]
        logger.info("Dry run: would process %d videos. Sample IDs: %s", len(to_process), sample_ids)
        print_summary(total_scanned, 0, skipped_existing, skipped_no_id)
        return 0

    # Process in batches
    for batch in chunked(to_process, args.batch_size):
        batch_ids = [b.get("video_id") for b in batch]
        try:
            ret = embedder.embed_videos(batch)
        except Exception:
            logger.exception("Embedding failed for batch starting with %s", batch_ids[:1])
            continue

        # Normalize return: accept:
        # - a list of EmbeddingResult objects (preferred)
        # - a tuple (embeddings, texts)
        # - embeddings array/list only
        embeddings: List[Any]
        texts: List[str]
        videos_to_add: List[Dict[str, Any]] = list(batch)

        if isinstance(ret, tuple) and len(ret) == 2:
            embeddings, texts = ret
        elif isinstance(ret, list) and len(ret) > 0 and hasattr(ret[0], 'embedding'):
            # Handle list[EmbeddingResult]
            id_to_result = {r.video_id: r for r in ret if getattr(r, 'success', True)}
            embeddings = []
            texts = []
            kept_videos: List[Dict[str, Any]] = []
            for v in batch:
                vid = v.get('video_id')
                res = id_to_result.get(vid)
                if res is None or getattr(res.embedding, 'size', None) == 0:
                    logger.warning("Skipping video %s: no embedding produced", vid)
                    continue
                embeddings.append(res.embedding)
                texts.append(res.text or str(v.get('title') or v.get('description') or ""))
                kept_videos.append(v)
            videos_to_add = kept_videos
        else:
            embeddings = ret
            texts = [str(b.get("title") or b.get("description") or "") for b in batch]

        try:
            if videos_to_add:
                # If update flag is set, delete existing ids first to allow re-insert
                if args.update:
                    ids_to_delete: List[str] = []
                    for v in videos_to_add:
                        vid = v.get('video_id')
                        if not isinstance(vid, str):
                            continue
                        try:
                            if store.exists(vid):
                                ids_to_delete.append(vid)
                        except Exception:
                            logger.exception("Error checking existence for %s; skipping delete", vid)
                    if ids_to_delete:
                        try:
                            store.delete_videos(ids_to_delete)
                            logger.info("Deleted %d existing videos before update", len(ids_to_delete))
                        except Exception:
                            logger.exception("Failed to delete existing videos before update: %s", ids_to_delete)

                store.add_videos(videos_to_add, embeddings, texts)
                added_count = len(videos_to_add)
                total_added += added_count
                for v in videos_to_add[:5]:
                    vid = v.get('video_id')
                    if not isinstance(vid, str):
                        continue
                    try:
                        if store.exists(vid):
                            sample_added_ids.extend(vid)
                    except:
                        logger.exception("Faild to extract existing video: %s", vid)
                logger.info("Added %d vectors (example id: %s)", added_count, [v.get('video_id') for v in videos_to_add[:1]])
            else:
                logger.info("No videos to add for this batch")
        except Exception:
            logger.exception("Failed to add batch to store for ids %s", batch_ids)

    print_summary(total_scanned, total_added, skipped_existing, skipped_no_id, sample_added_ids)
    return 0


def print_summary(total_scanned: int, added: int, skipped_existing: int, skipped_no_id: int, sample_added_ids: Optional[List[str]] = None) -> None:
    sample_added_ids = sample_added_ids or []
    print("--- Embedding pipeline summary ---")
    print(f"Total scanned: {total_scanned}")
    print(f"Added: {added}")
    print(f"Skipped (existing): {skipped_existing}")
    print(f"Skipped (no id / invalid): {skipped_no_id}")
    if sample_added_ids:
        print(f"Sample added IDs: {sample_added_ids[:10]}")


if __name__ == "__main__":
    raise SystemExit(main())
