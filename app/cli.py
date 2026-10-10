"""Batch CLI: python -m app.cli ingest <folder> | reembed [--model NAME] | load | search <query>

`ingest` runs every file in a folder through the full pipeline and prints a summary;
`reembed` re-chunks and re-embeds existing documents under a model without re-normalising;
`load` rebuilds the relational tables from the normalised zone; `search` runs a
semantic query against embedded chunks.
"""

import argparse
import logging
import mimetypes
from pathlib import Path

import psycopg

from . import embed
from .db import DATABASE_URL, new_run
from .pipeline import PipelineError, load_all, process


def ingest_folder(conn, folder: Path) -> tuple[int, int, list[tuple[str, str]]]:
    """Returns (processed, skipped_duplicates, [(filename, reason)]) - one bad file never
    stops the batch."""
    run_id = new_run(conn, embedding_model=embed.MODEL)
    conn.commit()
    processed = skipped = 0
    failed = []
    for path in sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")):
        try:
            result = process(conn, path.read_bytes(), path.name,
                             mimetypes.guess_type(path.name)[0], run_id=run_id)
            if result["duplicate"]:
                skipped += 1
            else:
                processed += 1
        except PipelineError as exc:
            failed.append((path.name, exc.code))
        except Exception as exc:  # process() already rolled back and quarantined
            failed.append((path.name, type(exc).__name__))
    conn.execute(
        "UPDATE pipeline_runs SET files_new = %s, files_duplicate = %s, files_failed = %s,"
        " finished_at = now() WHERE id = %s",
        (processed, skipped, len(failed), run_id),
    )
    conn.commit()
    return processed, skipped, failed


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="ingest every file in a folder")
    ingest.add_argument("folder", type=Path)
    reembed = sub.add_parser("reembed", help="re-chunk and re-embed all documents")
    reembed.add_argument("--model", help=f"embedding model name (default: {embed.MODEL})")
    sub.add_parser("load", help="normalised zone -> relational tables")
    search = sub.add_parser("search", help="semantic search over embedded chunks")
    search.add_argument("query")
    search.add_argument("-k", "--top-k", type=int, default=5)
    search.add_argument("--model", help=f"embedding model name (default: {embed.MODEL})")
    search.add_argument("--type", dest="document_type", help="filter on documents.document_type")
    search.add_argument("--country", help="filter on documents.country_code")
    args = parser.parse_args()

    with psycopg.connect(DATABASE_URL) as conn:
        if args.command == "ingest":
            processed, skipped, failed = ingest_folder(conn, args.folder)
            print(f"processed {processed}, skipped {skipped} (duplicate), failed {len(failed)}")
            for name, reason in failed:
                print(f"  failed: {name}: {reason}")
        elif args.command == "reembed":
            done, failed = embed.embed_loaded(conn, model=args.model)
            print(f"re-embedded {done} document(s), failed {failed}")
        elif args.command == "search":
            # same boundary mask as POST /search; imported here so other commands don't
            # pay for the FastAPI import
            from .main import mask_nric
            if not args.query.strip():
                parser.error("empty query")
            vec = [float(x)
                   for x in next(embed.embedder(args.model or embed.MODEL).embed([args.query]))]
            rows = embed.search(conn, vec, model=args.model,
                                document_type=args.document_type, country_code=args.country,
                                top_k=args.top_k)
            for _cid, doc_id, dtype, page, score, text, x0, y0, x1, y1, model_name in rows:
                print(f"{score:.4f}  {doc_id}  {dtype} p{page}  "
                      f"[{x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f}]  {model_name}  {mask_nric(text)}")
        else:
            loaded, skipped, failed = load_all(conn)
            print(f"loaded {loaded}, skipped {skipped} (already in documents), failed {failed}")


if __name__ == "__main__":
    main()
