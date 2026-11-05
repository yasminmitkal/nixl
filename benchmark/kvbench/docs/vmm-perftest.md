# VMM Perftest

This refers to the command `vmm-perftest`.

## Overview
Measures the cost of keeping a peer up to date while one CUDA VMM buffer changes its mapping. Rank 0 (the owner) reserves one virtual-address range of `chunks` chunks and changes it one chunk per step. After each change it calls `register_memory` again and builds new metadata. Rank 1 (the peer) loads that metadata with `add_remote_agent` and transfers to the chunk that changed.

Every call is timed on its own, for every step:

| Timing | Call | Rank |
|---|---|---|
| `reserve` | `cuMemAddressReserve`, step 0 only | owner |
| `map` | this step's `cuMemMap`, `cuMemUnmap`, or remap | owner |
| `register` | `register_memory` | owner |
| `metadata` | `get_partial_agent_metadata` of the whole reserve, or `get_agent_metadata` | owner |
| `load` | `add_remote_agent` of that metadata | peer |
| `write_prep`, `read_prep` | `initialize_xfer` | peer |
| `write_xfer`, `read_xfer` | the first transfer after the change, posted until `DONE` | peer |
| `write_steady_xfer`, `read_steady_xfer` | `xfer_iters` reposts of the same request | peer |
| `span_read_*` | one READ over the run of mapped chunks around the target | peer |
| `hole_write_*`, `hole_read_*` | `shrink` only: a transfer into the chunk just unmapped | peer |
| `hole_span_read_*` | `shrink` only: a READ from the middle of the last mapped chunk into the hole | peer |

Patterns:
- **grow**: start with chunk 0, map 1, 2, ... up to the last chunk
- **shrink**: start with every chunk, unmap the last one until only chunk 0 is left
- **ends**: start with the first and last chunk, map the middle from both ends inward
- **remap**: start with every chunk, replace each chunk's backing one per step

Step 0 always registers the whole reserve. On later steps `register: full` registers the whole reserve again, and `register: partial` registers only the changed chunk.

## Failures
A NIXL call that raises, or a transfer that ends in an error, fails the run. The error names the configuration, the step, and the call. Transfers into a hole are the exception: they are timed until they complete or return an error, whichever happens. A transfer that does not finish within `transfer_timeout_sec` fails the run.

With `--verify-buffers`, every non-hole transfer also checks the first and last 8 bytes of each chunk it touched, and a mismatch fails the run. The checks run outside the timers.

## Reports
For every configuration, the p50 of each call per step is logged as a table, in milliseconds. A column is shown only when some step has that timing:

```
  Step  Action    Chunks    Blob (B)    Map    Register    Metadata    Load    WRITE prep    WRITE first    WRITE steady    ...
```

With `--json-output-path`, rank 0 writes every configuration with, per step and per call, `avg`, `p50`, `p90`, `p99`, `min`, and `max` in milliseconds, computed as in `sequential-ct-perftest`. Each step also has `blob_bytes`, `blob_changed`, and `owner_free_change_bytes`. The file is rewritten after every configuration, so a run that crashes keeps the ones that finished.

With `--report-transport`, the peer sets `UCX_PROTO_INFO=y`, captures its own output, and adds the UCX protocol and transport each transfer used to the JSON, for example `zero-copy cuda_ipc/cuda` or `software emulation tcp/lo`.

## Usage
The YAML file holds the sweep. Every key except `path` has the default shown:

```yaml
path: nvlink            # nvlink: fabric handles; rdma: POSIX fd handles
patterns: [grow, shrink, ends, remap]
register: [full, partial]
metadata: [partial, agent]
ops: [WRITE, READ]
chunk_units: [1, 4, 16] # chunk size in units of the CUDA allocation granularity
chunks: 8               # chunks in the reserve
payload: 4096           # bytes per WRITE and READ, 8 up to one chunk
iters: 5                # measured passes of each configuration
warmup_iters: 1         # passes run first and discarded
xfer_iters: 10          # reposts of each transfer after the first one
transfer_timeout_sec: 15
exchange_timeout_sec: 300
```

`path` selects only the CUDA handle type. The transports come from `UCX_TLS`, which the launcher sets. Every pass uses fresh agents and a fresh buffer in the same two processes, so state UCX caches per process can carry over from one pass to the next.

The command runs on exactly 2 ranks, coordinated through etcd like the CT perftests. Each rank uses device 0 of what it sees, so give each rank its own GPU with `CUDA_VISIBLE_DEVICES`. GPU memory is allocated with the CUDA driver API from `cuda-python`.

```bash
export NIXL_ETCD_ENDPOINTS=http://<etcd-host>:2379
export NIXL_ETCD_NAMESPACE="/nixl/kvbench/$(uuidgen)"
export UCX_TLS=cuda_ipc,cuda_copy,tcp

# One node, two GPUs
CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=2 python main.py vmm-perftest examples/vmm-perftest.yaml --json-output-path ./vmm.json &
CUDA_VISIBLE_DEVICES=1 RANK=1 WORLD_SIZE=2 python main.py vmm-perftest examples/vmm-perftest.yaml

# Or with slurm
srun -n 2 --gpus-per-task=1 python main.py vmm-perftest examples/vmm-perftest.yaml --json-output-path ./vmm.json
```

The owner and the peer exchange requests through etcd between calls. Those exchanges are outside every timer.

## Implementation Details
- `test/vmm_dynamic_buffer_perftest.py` implements the patterns, the owner and peer roles, and the report
- `test/test_vmm_perftest_config.py` checks the patterns, the configuration, and the owner and peer flow without a GPU
