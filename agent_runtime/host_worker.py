"""Detached local host entry point."""
import argparse
import asyncio
import signal
from pathlib import Path

from agent_runtime.host import BackgroundHost


async def serve(database: Path, *, concurrency: int = 4) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    async with BackgroundHost(database, concurrency=concurrency) as host:
        await host.serve(stop=stop, idle_timeout=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Own background Praxis execution in a Harness database")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    try:
        asyncio.run(serve(args.database, concurrency=args.concurrency))
    except RuntimeError as error:
        if str(error) != "a background host already owns this database":
            raise


if __name__ == "__main__":
    main()
