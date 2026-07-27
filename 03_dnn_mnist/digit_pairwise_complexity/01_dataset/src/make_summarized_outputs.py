from __future__ import annotations

from utils.summarized_outputs import build_summarized_outputs


def main() -> None:
    counts = build_summarized_outputs()
    print(
        "summarized_outputs "
        f"dataset_summary_rows={counts['dataset_summary_rows']} "
        f"sample_index_rows={counts['sample_index_rows']}"
    )


if __name__ == "__main__":
    main()
