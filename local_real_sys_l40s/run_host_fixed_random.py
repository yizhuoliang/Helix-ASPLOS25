import argparse
import os
import time

import llm_host

from llm_sys.utils import CONFIG_BROADCAST_ADDR, get_local_ip, FlyingQuery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "real_sys_config.txt"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "result_fixed_random"))
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--init-wait-s", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    host_ip = get_local_ip()
    llm_host.start_network_threads(CONFIG_BROADCAST_ADDR, host_ip, args.config, "random")
    time.sleep(args.init_wait_s)
    print("[Python] Cluster initialization finished!")

    ground_zero = time.time()
    next_query_id = 0
    flying_queries = {}
    query_routes = []
    events = []

    for _ in range(args.num_requests):
        qid = next_query_id
        next_query_id += 1
        llm_host.launch_request(
            "prompt",
            qid,
            args.input_len,
            args.input_len + args.output_len,
            [i for i in range(args.input_len)],
            False,
            [],
            [],
            [],
        )
        flying_queries[qid] = FlyingQuery(
            query_uid=qid,
            input_length=args.input_len,
            output_length=args.output_len,
            compute_node_uids=None,
            start_layers=None,
            end_layers=None,
            pipeline=None,
        )
        now = time.time() - ground_zero
        events.append((now, qid, "out", "prompt", 0, args.input_len + 1))
        print(f"Send out query {qid}, input_len={args.input_len}, max_len={args.input_len + args.output_len}")

    while True:
        now = time.time() - ground_zero
        if now > args.timeout_s:
            break
        if not flying_queries:
            break

        finished_query_ids, _generated_token_ids, routes, num_layers = llm_host.gather_finished_requests()
        for query_uid, route_list, num_layer_list in zip(finished_query_ids, routes, num_layers):
            q = flying_queries.get(query_uid)
            if q is None:
                continue

            if q.processed_tokens == 0:
                events.append((now, query_uid, "in", "prompt", 0, q.input_length + 1))
                q.processed_tokens += q.input_length + 1

                cur_log_start = 0
                start_layer_ids, end_layer_ids = [], []
                for nl in num_layer_list:
                    start_layer_ids.append(cur_log_start)
                    end_layer_ids.append(cur_log_start + nl)
                    cur_log_start += nl
                query_routes.append(
                    (query_uid, q.input_length, q.output_length, route_list, start_layer_ids, end_layer_ids)
                )
                print(f"Query {query_uid} prompt finished; route={route_list}, layer_nums={num_layer_list}")
            else:
                events.append((now, query_uid, "in", "decode", q.processed_tokens, 1))
                q.processed_tokens += 1

            max_size = q.input_length + q.output_length
            if q.processed_tokens >= max_size:
                del flying_queries[query_uid]
                print(f"Query {query_uid} finished (total_len={q.processed_tokens})")
                continue

            llm_host.launch_request(
                "decode",
                query_uid,
                q.processed_tokens,
                max_size,
                [-1],
                False,
                [],
                [],
                [],
            )
            events.append((now, query_uid, "out", "decode", q.processed_tokens, 1))

        time.sleep(0.01)

    query_routes_file = os.path.join(args.out, "query_route.txt")
    events_file = os.path.join(args.out, "events.txt")
    with open(query_routes_file, "w") as f:
        for item in query_routes:
            f.write(f"{item}\n")
    with open(events_file, "w") as f:
        for item in events:
            f.write(f"{item}\n")

    os._exit(0)


if __name__ == "__main__":
    main()
