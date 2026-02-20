import argparse
import os

from llm_sys.heuristic_host import run_heuristic_host_offline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "real_sys_config.txt"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "result_offline_random"))
    parser.add_argument("--initial-launch", type=int, default=4)
    parser.add_argument("--duration", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_heuristic_host_offline(
        scheduler_name="random",
        real_sys_config_file_name=args.config,
        initial_launch_num=args.initial_launch,
        duration=args.duration,
        result_logging_dir=args.out,
    )


if __name__ == "__main__":
    main()
