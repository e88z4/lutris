"""Multi-connection parallel downloader for GOG game files.

Uses HTTP Range requests to download different byte ranges of a file
simultaneously across multiple threads, significantly improving download
speeds for large GOG installer files.

This downloader is a drop-in replacement for the standard Downloader class,
maintaining API compatibility with DownloadProgressBox and
DownloadCollectionProgressBox.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

from lutris import __version__
from lutris.util import jobs
from lutris.util.downloader import DEFAULT_CHUNK_SIZE, Downloader, get_time
from lutris.util.log import logger


class GOGDownloader(Downloader):
    """Multi-connection parallel downloader optimized for GOG CDN downloads.

    Downloads large files using multiple simultaneous HTTP Range requests,
    each writing to a different region of the output file. Falls back to
    single-stream download if the server doesn't support Range requests
    or the file is too small to benefit from parallelism.

    Designed to be API-compatible with Downloader so it works seamlessly
    with DownloadProgressBox and DownloadCollectionProgressBox.
    """

    DEFAULT_WORKERS = 4
    MIN_CHUNK_SIZE = 5 * 1024 * 1024  # 5MB minimum per worker
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2  # seconds between retries

    def __init__(
        self,
        url: str,
        dest: str,
        overwrite: bool = False,
        referer: Optional[str] = None,
        cookies: Any = None,
        headers: Dict[str, str] = None,
        session: Optional[requests.Session] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        num_workers: int = DEFAULT_WORKERS,
    ) -> None:
        super().__init__(
            url=url,
            dest=dest,
            overwrite=overwrite,
            referer=referer,
            cookies=cookies,
            headers=headers,
            session=session,
            chunk_size=chunk_size,
        )
        self.num_workers = max(1, num_workers)
        self._download_lock = threading.Lock()
        # Create a dedicated session with connection pooling sized for our workers
        self._parallel_session = requests.Session()
        adapter = HTTPAdapter(pool_maxsize=self.num_workers + 2)
        self._parallel_session.mount("https://", adapter)
        self._parallel_session.mount("http://", adapter)
        self._parallel_session.headers["User-Agent"] = "Lutris/%s" % __version__

    def __repr__(self):
        return "GOG parallel downloader (%d workers) for %s" % (self.num_workers, self.url)

    def start(self):
        """Start parallel download job."""
        logger.debug("⬇ GOG parallel (%d workers): %s", self.num_workers, self.url)
        self.state = self.DOWNLOADING
        self.last_check_time = get_time()
        if self.overwrite and os.path.isfile(self.dest):
            os.remove(self.dest)
        # Workers manage their own file I/O - no shared file_pointer needed
        self.file_pointer = None
        self.thread = jobs.AsyncCall(self.async_download, None)
        self.stop_request = self.thread.stop_request

    def cancel(self):
        """Request download stop and remove destination file."""
        logger.debug("❌ GOG parallel: %s", self.url)
        self.state = self.CANCELLED
        if self.stop_request:
            self.stop_request.set()
        # No shared file_pointer to close - workers handle their own
        if os.path.isfile(self.dest):
            os.remove(self.dest)

    def on_download_completed(self):
        """Mark download as complete."""
        if self.state == self.CANCELLED:
            return
        logger.debug("✅ GOG parallel download finished: %s", self.url)
        if not self.downloaded_size:
            logger.warning("Downloaded file is empty")
        if not self.full_size:
            self.progress_fraction = 1.0
            self.progress_percentage = 100
        self.state = self.COMPLETED
        # No shared file_pointer to close

    def _build_request_headers(self) -> dict:
        """Build HTTP headers for download requests."""
        headers = requests.utils.default_headers()
        headers["User-Agent"] = "Lutris/%s" % __version__
        if self.referer:
            headers["Referer"] = self.referer
        if self.headers:
            for key, value in self.headers.items():
                headers[key] = value
        return headers

    def _calculate_ranges(self, file_size: int) -> List[Tuple[int, int]]:
        """Split file into byte ranges for parallel download.

        Returns a list of (start, end) tuples representing inclusive byte ranges.
        """
        chunk_size = file_size // self.num_workers
        ranges = []
        for i in range(self.num_workers):
            start = i * chunk_size
            end = file_size - 1 if i == self.num_workers - 1 else (i + 1) * chunk_size - 1
            ranges.append((start, end))
        return ranges

    def async_download(self):
        """Execute multi-connection parallel download."""
        try:
            headers = self._build_request_headers()

            # Step 1: Resolve URL (follow redirects) and check capabilities
            final_url, file_size, supports_range = self._probe_server(headers)
            self.full_size = file_size

            # Fall back to single-stream if Range not supported or file too small
            if not supports_range or file_size < self.MIN_CHUNK_SIZE * 2:
                logger.info(
                    "GOG download: falling back to single-stream "
                    "(range=%s, size=%d bytes)",
                    supports_range,
                    file_size,
                )
                self._single_stream_download(final_url, headers)
                return

            self.progress_event.set()  # Signal that size is known

            # Step 2: Pre-allocate output file
            with open(self.dest, "wb") as f:
                f.truncate(file_size)

            # Step 3: Calculate byte ranges for workers
            ranges = self._calculate_ranges(file_size)
            per_worker_mb = (file_size // self.num_workers) // (1024 * 1024)
            logger.info(
                "GOG parallel download: %d workers, %d MB total, ~%d MB/worker",
                self.num_workers,
                file_size // (1024 * 1024),
                per_worker_mb,
            )

            # Step 4: Download chunks in parallel
            errors = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_range = {}
                for start, end in ranges:
                    future = executor.submit(
                        self._download_range, final_url, headers, start, end
                    )
                    future_to_range[future] = (start, end)

                for future in as_completed(future_to_range):
                    try:
                        future.result()
                    except Exception as ex:
                        rng = future_to_range[future]
                        logger.error(
                            "Worker failed for range %d-%d: %s", rng[0], rng[1], ex
                        )
                        errors.append(ex)
                        # Signal other workers to stop
                        if self.stop_request:
                            self.stop_request.set()

            if errors:
                raise errors[0]

            self.on_download_completed()
        except Exception as ex:
            logger.exception("GOG parallel download failed: %s", ex)
            self.on_download_failed(ex)

    def _probe_server(self, headers: dict) -> Tuple[str, int, bool]:
        """Probe the server to determine final URL, file size, and Range support.

        Uses a HEAD request to follow redirects (e.g., GOG API → CDN URL),
        get Content-Length, and check Accept-Ranges header.

        Returns:
            Tuple of (final_url, file_size, supports_range)
        """
        resp = self._parallel_session.head(
            self.url, headers=headers, allow_redirects=True, timeout=30, cookies=self.cookies
        )
        resp.raise_for_status()

        final_url = resp.url
        file_size = int(resp.headers.get("Content-Length", 0))
        accept_ranges = resp.headers.get("Accept-Ranges", "")
        supports_range = "bytes" in accept_ranges.lower()

        # Some servers don't advertise Accept-Ranges but still support it.
        # If we got a Content-Length, try a small Range request to verify.
        if file_size and not supports_range:
            supports_range = self._test_range_support(final_url, headers)

        logger.debug(
            "GOG probe: url=%s, size=%d, range=%s",
            final_url[:80],
            file_size,
            supports_range,
        )
        return final_url, file_size, supports_range

    def _test_range_support(self, url: str, headers: dict) -> bool:
        """Test if server actually supports Range requests with a small probe."""
        try:
            test_headers = dict(headers)
            test_headers["Range"] = "bytes=0-0"
            resp = self._parallel_session.get(
                url, headers=test_headers, stream=True, timeout=10, cookies=self.cookies
            )
            resp.close()
            return resp.status_code == 206
        except Exception:
            return False

    def _download_range(self, url: str, headers: dict, start: int, end: int) -> None:
        """Download a specific byte range and write to the correct file offset.

        Each worker opens its own file handle, seeks to its starting offset,
        and writes sequentially from there. Since workers operate on
        non-overlapping regions, no file-level locking is needed.

        Retries up to RETRY_ATTEMPTS times with exponential backoff.
        """
        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                range_headers = dict(headers)
                range_headers["Range"] = "bytes=%d-%d" % (start, end)

                response = self._parallel_session.get(
                    url,
                    headers=range_headers,
                    stream=True,
                    timeout=30,
                    cookies=self.cookies,
                )

                if response.status_code not in (200, 206):
                    raise requests.HTTPError(
                        "HTTP %d for range %d-%d" % (response.status_code, start, end),
                        response=response,
                    )

                # If server returned 200 (ignoring Range), only write our portion
                if response.status_code == 200:
                    logger.warning(
                        "Server ignored Range header, reading full response "
                        "for range %d-%d",
                        start,
                        end,
                    )
                    self._write_from_full_response(response, start, end)
                    return

                # Normal 206 Partial Content response
                with open(self.dest, "r+b") as f:
                    f.seek(start)
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if self.stop_request and self.stop_request.is_set():
                            return
                        if chunk:
                            f.write(chunk)
                            with self._download_lock:
                                self.downloaded_size += len(chunk)
                            self.progress_event.set()
                return  # Success

            except Exception as ex:
                if self.stop_request and self.stop_request.is_set():
                    return  # Cancelled, don't retry
                if attempt < self.RETRY_ATTEMPTS - 1:
                    wait = self.RETRY_DELAY * (attempt + 1)
                    logger.warning(
                        "GOG range %d-%d attempt %d/%d failed: %s, retrying in %ds...",
                        start,
                        end,
                        attempt + 1,
                        self.RETRY_ATTEMPTS,
                        ex,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    raise

    def _write_from_full_response(
        self, response: requests.Response, start: int, end: int
    ) -> None:
        """Handle the case where server returns 200 instead of 206.

        Read the full response but only write our byte range portion.
        This is a fallback for non-compliant servers.
        """
        bytes_read = 0
        expected_size = end - start + 1
        with open(self.dest, "r+b") as f:
            f.seek(start)
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if self.stop_request and self.stop_request.is_set():
                    return
                if not chunk:
                    continue

                # Only write the portion that falls within our range
                chunk_start = bytes_read
                chunk_end = bytes_read + len(chunk)

                if chunk_end <= start:
                    # Before our range, skip
                    bytes_read += len(chunk)
                    continue
                elif chunk_start >= end + 1:
                    # Past our range, done
                    break
                else:
                    # Calculate the slice of this chunk we need
                    slice_start = max(0, start - chunk_start)
                    slice_end = min(len(chunk), end + 1 - chunk_start)
                    data = chunk[slice_start:slice_end]
                    f.write(data)
                    with self._download_lock:
                        self.downloaded_size += len(data)
                    self.progress_event.set()

                bytes_read += len(chunk)
                if bytes_read >= end + 1:
                    break

    def _single_stream_download(self, url: str, headers: dict) -> None:
        """Fallback single-stream download when Range requests aren't supported.

        Uses the parallel session for connection pooling benefits.
        """
        response = self._parallel_session.get(
            url, headers=headers, stream=True, timeout=30, cookies=self.cookies
        )
        response.raise_for_status()
        self.full_size = int(response.headers.get("Content-Length", "").strip() or 0)
        self.progress_event.set()

        with open(self.dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if self.stop_request and self.stop_request.is_set():
                    break
                if chunk:
                    self.downloaded_size += len(chunk)
                    f.write(chunk)
                self.progress_event.set()

        self.on_download_completed()
