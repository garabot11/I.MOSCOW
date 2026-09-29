# results/ (not part of the image)

| directory | what |
|---|---|
| `dev1` … `dev9` | development iterations on the host (every 4th frame, ROS Jazzy Python, older code versions) – kept as the tuning history referred to in `docs/experiments.md` |
| `final_offline` | first full-resolution run in the Humble container (code before the last two rule changes) |
| `final_offline_v2` | **final full-resolution run** in the Humble container: one JSONL per bag, `*_summary.json`, `metrics.json` from `scripts/evaluate.py` |
| `ablation/<variant>` | experiment matrix (every 2nd frame) and `report.md` |
| `humble` | smoke/timing runs and the streaming tests (`stream_*/stream.jsonl`, `node.log`, `play.log`) |
| `clean_check` | offline run of the final image built from a clean copy of the sources |
| `demo_*` | logs and RViz configs written by `scripts/run_demo.sh` |
| `final_offline_v3` | full-resolution run of the corrected image (after the audit `ПРОВЕРКА_СООТВЕТСТВИЯ_PDF.md`): same records as `final_offline_v2` (compared frame by frame), summaries with the completeness check |
| `final_offline_v4` | full-resolution run of the final image `sha256:69acee4e…` (raw CDR decoding, fused prepare stage): identical frame by frame to `final_offline_v2` |
| `recheck/perf` | CDR parser equality on all 2488 messages, micro-benchmarks of decoding and the prepare stage |
| `recheck/load` | node CPU / RSS / throughput in a 10 Hz stream on P-cores and on the E-cores (proxy for older cores); `image_fix2/` = same code, incl. single-core pinning and decimation runs |
| `recheck/offline_cap180k`, `offline_fix2` | full runs with `max_analysed_points=180000` (recall drops to 12/47) and of the pre-final code (0 differences) |
| `recheck/mcap` | MCAP copy of `roundT_doubleT` (made with `convert.yaml`, deleted afterwards): offline records identical to SQLite |
| `recheck/host_play` | bag played outside the node container (second Humble container, host Jazzy) with/without the UDP-only Fast DDS profile |
| `recheck/stream_final` | whole-bag stream tests of the final image with `input_topic:=auto`, six bags + the MCAP copy |
| `recheck` | verification of the corrections: stream tests over whole bags with `coverage.json` (sent / processed / published by header stamp), README commands reproduced literally, failure probes of the batch scripts, RViz demo screenshot, `verification.json` (summary of all checks) |

Offline JSONL records (one per frame; `frame_index` = 0-based index in the bag): bag, topic, frame index, bag time, relative time, header time,
status, distance, temporal reset flag, processing/read time and the full detector output
(candidates, rejected clusters with reasons, rail-level and track-curve coefficients, timings).

Streaming logs of the node (`stream.jsonl`, `live.jsonl`, `demo.jsonl`) use `frame_index` = number
of the cloud *received* by the node (from 1); match them with offline results and annotations by
`header_time_ns` (`bag_tools coverage` does that).
