"""uv run python -m vega.web [--port N]"""

from __future__ import annotations

import sys

from vega.web.server import DEFAULT_PORT, serve


def main() -> None:
    port = DEFAULT_PORT
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    serve(port)


if __name__ == "__main__":
    main()
