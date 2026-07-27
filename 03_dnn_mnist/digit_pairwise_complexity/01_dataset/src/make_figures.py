from __future__ import annotations

from utils.figure_builders import build_figures


def main() -> None:
    path = build_figures()
    print(path)


if __name__ == "__main__":
    main()
