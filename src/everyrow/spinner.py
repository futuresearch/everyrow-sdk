import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Spinner:
    """An async spinner that displays a subtle animation during long operations."""
    FRAME_DELAY_s = 0.08

    def __init__(self, message: str = ""):
        self.message = message
        self._task: asyncio.Task[None] | None = None
        self._stop = False

    async def _spin(self) -> None:
        frame_idx = 0
        while not self._stop:
            frame = SPINNER_FRAMES[frame_idx % len(SPINNER_FRAMES)]
            text = f"\r{frame} {self.message}"
            sys.stdout.write(text)
            sys.stdout.flush()
            frame_idx += 1
            await asyncio.sleep(self.FRAME_DELAY_s)

    def start(self) -> None:
        self._stop = False
        self._task = asyncio.create_task(self._spin())

    async def stop(self, final_message: str | None = None) -> None:
        self._stop = True
        if self._task:
            await self._task
            self._task = None

        clear_width = len(self.message) + 3
        sys.stdout.write("\r" + " " * clear_width + "\r")
        if final_message:
            print(final_message)
        sys.stdout.flush()


@asynccontextmanager
async def spinner(message: str = "") -> AsyncIterator[Spinner]:
    """Context manager for displaying a spinner during async operations.

    Usage:
        async with spinner("Processing..."):
            await some_long_operation()
    """
    s = Spinner(message)
    s.start()
    try:
        yield s
    finally:
        await s.stop()
