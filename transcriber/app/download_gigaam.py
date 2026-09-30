"""Resume a pinned public GigaAM weight download using validated HTTP ranges.

Run inside the transcriber image with its /models volume mounted. Requires
Python and httpx only. Signed redirect URLs and response bodies are never logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
import random
import sys
import time
import uuid

import httpx


REVISION = "3905cd51c3ed4e88c8edf33f3302969ba480a327"
EXPECTED_SIZE = 2_341_592_643
EXPECTED_SHA256 = "c3fabefb50b41f08f4d7ad44e02c26c37d242882704cdcca2ebd98e45eff73d1"
URL = (
    "https://huggingface.co/ai-sage/GigaAM-Multilingual/resolve/"
    f"{REVISION}/pytorch_model.bin"
)
CHUNK_SIZE = 8 * 1024 * 1024
CONCURRENCY = 12
MAX_ATTEMPTS = 20
PART_COUNT = (EXPECTED_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE
CACHE_ROOT = Path(os.getenv("WHISPER_DOWNLOAD_ROOT", "/models")) / "gigaam"
PARTS_DIR = CACHE_ROOT / "range-download"
REPOSITORY_DIR = CACHE_ROOT / "models--ai-sage--GigaAM-Multilingual"
BLOB_PATH = REPOSITORY_DIR / "blobs" / EXPECTED_SHA256
SNAPSHOT_PATH = REPOSITORY_DIR / "snapshots" / REVISION / "pytorch_model.bin"


class RangeValidationError(Exception):
    pass


class ChunkDownloadError(Exception):
    pass


class ChecksumMismatchError(Exception):
    pass


class SnapshotConflictError(Exception):
    pass


def part_bounds(index: int) -> tuple[int, int, int]:
    start = index * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, EXPECTED_SIZE) - 1
    return start, end, end - start + 1


def part_path(index: int) -> Path:
    return PARTS_DIR / f"{index}.part"


def valid_part(index: int) -> bool:
    path = part_path(index)
    return path.is_file() and path.stat().st_size == part_bounds(index)[2]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ensure_snapshot() -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(SNAPSHOT_PATH):
        SNAPSHOT_PATH.symlink_to(f"../../blobs/{EXPECTED_SHA256}")
    elif SNAPSHOT_PATH.is_symlink():
        if SNAPSHOT_PATH.resolve() != BLOB_PATH.resolve():
            raise SnapshotConflictError()
    elif not (
        SNAPSHOT_PATH.is_file()
        and SNAPSHOT_PATH.stat().st_size == EXPECTED_SIZE
        and file_digest(SNAPSHOT_PATH) == EXPECTED_SHA256
    ):
        # Preserve anything already cached at the snapshot path.
        raise SnapshotConflictError()


async def download_part(client: httpx.AsyncClient, index: int) -> None:
    start, end, length = part_bounds(index)
    expected_range = f"bytes {start}-{end}/{EXPECTED_SIZE}"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        temporary = PARTS_DIR / f"{index}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            # Always request the canonical pinned URL again, so retrying can
            # obtain a fresh signed redirect instead of reusing an expired one.
            async with client.stream(
                "GET",
                URL,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                    "Cache-Control": "no-cache",
                },
            ) as response:
                if (
                    response.status_code != 206
                    or response.headers.get("Content-Range") != expected_range
                    or response.headers.get("Content-Encoding", "identity")
                    not in ("", "identity")
                ):
                    raise RangeValidationError()
                content_length = response.headers.get("Content-Length")
                if content_length is not None and content_length != str(length):
                    raise RangeValidationError()
                received = 0
                with temporary.open("xb") as target:
                    async for block in response.aiter_raw(chunk_size=256 * 1024):
                        received += len(block)
                        if received > length:
                            raise RangeValidationError()
                        target.write(block)
                    if received != length:
                        raise RangeValidationError()
                    target.flush()
                    os.fsync(target.fileno())
            # Partial bytes never become a resumable .part file.
            os.replace(temporary, part_path(index))
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Exception messages can contain signed URLs: log class names only.
            print(
                f"retry part={index} attempt={attempt}/{MAX_ATTEMPTS} "
                f"error={type(exc).__name__}",
                flush=True,
            )
            if attempt == MAX_ATTEMPTS:
                raise ChunkDownloadError() from None
            await asyncio.sleep(min(15.0, 0.75 * (2 ** (attempt - 1))) + random.random())
        finally:
            temporary.unlink(missing_ok=True)


def assemble_and_verify() -> None:
    BLOB_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = BLOB_PATH.with_name(f"{EXPECTED_SHA256}.{os.getpid()}.temp")
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("wb") as target:
            for index in range(PART_COUNT):
                if not valid_part(index):
                    raise RangeValidationError()
                with part_path(index).open("rb") as source:
                    while block := source.read(1024 * 1024):
                        target.write(block)
                        digest.update(block)
                        total += len(block)
            target.flush()
            os.fsync(target.fileno())
        if total != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
            raise ChecksumMismatchError()
        # Never publish unverified weights into the Hugging Face cache.
        os.replace(temporary, BLOB_PATH)
    finally:
        temporary.unlink(missing_ok=True)


async def main() -> None:
    started = time.monotonic()
    if (
        BLOB_PATH.is_file()
        and BLOB_PATH.stat().st_size == EXPECTED_SIZE
        and file_digest(BLOB_PATH) == EXPECTED_SHA256
    ):
        ensure_snapshot()
        print(f"verified cached blob bytes={EXPECTED_SIZE} sha256={EXPECTED_SHA256}", flush=True)
        return

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[int] = asyncio.Queue()
    complete = 0
    completed_bytes = 0
    for index in range(PART_COUNT):
        if valid_part(index):
            complete += 1
            completed_bytes += part_bounds(index)[2]
        else:
            queue.put_nowait(index)
    print(
        f"resume parts={complete}/{PART_COUNT} bytes={completed_bytes}/{EXPECTED_SIZE} "
        f"concurrency={CONCURRENCY}",
        flush=True,
    )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=20.0, read=45.0, write=45.0, pool=20.0),
        limits=httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY),
        follow_redirects=True,
    ) as client:
        async def worker() -> None:
            nonlocal complete, completed_bytes
            while True:
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await download_part(client, index)
                    complete += 1
                    completed_bytes += part_bounds(index)[2]
                    if complete % 6 == 0 or complete == PART_COUNT:
                        print(
                            f"progress parts={complete}/{PART_COUNT} "
                            f"bytes={completed_bytes}/{EXPECTED_SIZE} "
                            f"elapsed={time.monotonic() - started:.1f}s",
                            flush=True,
                        )
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(CONCURRENCY)]
        try:
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    print("assembling and verifying SHA-256", flush=True)
    assemble_and_verify()
    ensure_snapshot()
    print(
        f"verified bytes={EXPECTED_SIZE} sha256={EXPECTED_SHA256} "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("stopped error=KeyboardInterrupt", flush=True)
        sys.exit(130)
    except Exception as exc:
        print(f"failed error={type(exc).__name__}", flush=True)
        sys.exit(1)
