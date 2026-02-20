import argparse
import os

from llm_sys.worker import run_worker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "artifact_evaluation",
            "single_cluster",
            "models",
            "llama30b",
        ),
    )
    parser.add_argument("--vram-usage", type=float, default=0.65)
    args = parser.parse_args()

    run_worker(scheduling_method="random", model_name=args.model, vram_usage=args.vram_usage)


if __name__ == "__main__":
    main()
