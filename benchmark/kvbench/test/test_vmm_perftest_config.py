# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vmm-perftest checks that need no GPU and no second rank."""

import json
import os
import subprocess
import sys
import threading

import pytest

p = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, p + "/../")

import test.vmm_dynamic_buffer_perftest as vmm  # noqa: E402

G = 2 << 20


def _shape(steps):
    return [(step.action, step.chunks, step.target) for step in steps]


def test_grow_plan():
    steps = vmm.plan("grow", 4)
    assert _shape(steps) == [
        ("map", (0,), 0),
        ("map", (1,), 1),
        ("map", (2,), 2),
        ("map", (3,), 3),
    ]
    assert all(step.hole is None for step in steps)


def test_shrink_plan():
    steps = vmm.plan("shrink", 4)
    assert _shape(steps) == [
        ("map", (0, 1, 2, 3), 0),
        ("unmap", (3,), 2),
        ("unmap", (2,), 1),
        ("unmap", (1,), 0),
    ]
    assert [step.hole for step in steps] == [None, 3, 2, 1]


def test_ends_plan():
    steps = vmm.plan("ends", 6)
    assert _shape(steps) == [
        ("map", (0, 5), 0),
        ("map", (1,), 1),
        ("map", (4,), 4),
        ("map", (2,), 2),
        ("map", (3,), 3),
    ]


def test_remap_plan():
    steps = vmm.plan("remap", 3)
    assert _shape(steps) == [
        ("map", (0, 1, 2), 0),
        ("remap", (0,), 0),
        ("remap", (1,), 1),
        ("remap", (2,), 2),
    ]


def test_plan_rejects_one_chunk_and_unknown_pattern():
    with pytest.raises(ValueError):
        vmm.plan("grow", 1)
    with pytest.raises(ValueError):
        vmm.plan("delta", 4)


def test_span():
    assert [vmm.span(step) for step in vmm.plan("grow", 4)] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
    ]
    assert [vmm.span(step) for step in vmm.plan("ends", 6)] == [
        (0, 1),
        (0, 2),
        (4, 2),
        (0, 3),
        (0, 6),
    ]
    assert [vmm.span(step) for step in vmm.plan("shrink", 4)] == [
        (0, 4),
        (0, 3),
        (0, 2),
        (0, 1),
    ]


def test_step_info_span_only_with_read():
    grow = vmm.plan("grow", 4)
    both = _case(ops=("WRITE", "READ"))
    assert "span_chunks" not in vmm.step_info(both, 0, grow[0])
    assert vmm.step_info(both, 2, grow[2])["span_chunks"] == 3
    assert "span_chunks" not in vmm.step_info(_case(ops=("WRITE",)), 2, grow[2])
    shrink = vmm.plan("shrink", 4)
    assert vmm.step_info(both, 1, shrink[1])["hole_chunk"] == 3


def test_config_defaults_and_lists():
    config = vmm.VmmPerftestConfig.from_dict({"path": "rdma"})
    assert config.patterns == vmm.PATTERNS
    assert config.ops == ("WRITE", "READ")
    assert config.chunk_units == (1, 4, 16)
    assert (config.iters, config.warmup_iters, config.xfer_iters) == (5, 1, 10)

    config = vmm.VmmPerftestConfig.from_dict(
        {"path": "nvlink", "patterns": "grow,ends", "ops": ["read"], "chunk_units": 4}
    )
    assert config.patterns == ("grow", "ends")
    assert config.ops == ("READ",)
    assert config.chunk_units == (4,)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"path": "ksm"},
        {"path": "rdma", "patterns": ["delta"]},
        {"path": "rdma", "register": ["subrange"]},
        {"path": "rdma", "metadata": ["delta"]},
        {"path": "rdma", "ops": ["SEND"]},
        {"path": "rdma", "chunk_units": [0]},
        {"path": "rdma", "chunks": 1},
        {"path": "rdma", "payload": 4},
        {"path": "rdma", "iters": 0},
        {"path": "rdma", "warmup_iters": -1},
        {"path": "rdma", "repeats": 5},
    ],
)
def test_config_rejects(data):
    with pytest.raises(ValueError):
        vmm.VmmPerftestConfig.from_dict(data)


def test_example_config_loads():
    yaml = pytest.importorskip("yaml")
    example = os.path.join(p, "..", "examples", "vmm-perftest.yaml")
    with open(example) as f:
        config = vmm.VmmPerftestConfig.from_dict(yaml.safe_load(f))
    assert config == vmm.VmmPerftestConfig(path="nvlink")


def test_cases_sweep():
    config = vmm.VmmPerftestConfig.from_dict(
        {"path": "nvlink", "patterns": ["grow", "ends"], "chunk_units": [1, 4]}
    )
    found = vmm.cases(config, G)
    assert len(found) == 2 * 2 * 2 * 2
    assert {case.register for case in found} == {"full", "partial"}
    assert {case.metadata for case in found} == {"partial", "agent"}
    assert found[0].chunk == G and found[-1].reserved == 8 * 4 * G


def test_cases_reject_payload_above_chunk():
    config = vmm.VmmPerftestConfig.from_dict({"path": "rdma", "payload": G + 1})
    with pytest.raises(ValueError):
        vmm.cases(config, G)


def test_stats_ms():
    assert vmm.stats_ms([]) is None
    stats = vmm.stats_ms([3_000_000, 1_000_000, 2_000_000])
    assert stats == {
        "avg": 2.0,
        "p50": 2.0,
        "p90": 3.0,
        "p99": 3.0,
        "min": 1.0,
        "max": 3.0,
    }
    many = vmm.stats_ms(list(range(1, 201)))
    assert many is not None and many["p99"] == 199 / 1e6


# =============================================================================
# Owner pass, with the GPU and the peer replaced
# =============================================================================


def _case(**changes):
    base = dict(
        path="rdma",
        pattern="grow",
        register="partial",
        metadata="partial",
        ops=("WRITE", "READ"),
        chunk_units=1,
        chunks=4,
        granularity=G,
        payload=4096,
        xfer_iters=2,
    )
    return vmm.Case(**{**base, **changes})


class FakeChannel:
    broken = False

    def __init__(self, replies=None):
        self.sent = []
        self.replies = replies or {}

    def request(self, message):
        self.sent.append(message)
        reply = self.replies.get(message["cmd"], {})
        return reply(message) if callable(reply) else reply


class FakeAgent:
    def __init__(self):
        self.registered = []

    def reg_descs(self, start, length):
        return (start, length)

    def register(self, descs):
        self.registered.append(descs)

    def partial_metadata(self, descs):
        return b"blob-%d" % len(self.registered)

    def agent_metadata(self):
        return b"agent"

    def close(self):
        pass


@pytest.fixture
def no_gpu(monkeypatch):
    calls = []
    for name in ("map_chunk", "remap_chunk", "unmap_chunk", "memset"):
        monkeypatch.setattr(
            vmm, name, lambda *args, _name=name, **kw: calls.append((_name, args))
        )
    monkeypatch.setattr(vmm, "edges_match", lambda *args: True)
    monkeypatch.setattr(vmm, "free_bytes", lambda: 0)
    return calls


def _owner(case, replies=None, verify=False):
    owner = vmm.OwnerPass(case, FakeChannel(replies), 0, verify, False, 1.0)
    owner.arena = vmm.Arena(0, 1 << 40, case.reserved, G, "posix")
    owner.agent = FakeAgent()
    owner.where = "here"
    return owner


def _done(message):
    return {"state": "DONE", "prep_ns": 1, "xfer_ns": 2, "steady": [3, 4]}


def test_register_ranges(no_gpu):
    case = _case(register="partial")
    owner = _owner(case)
    steps = vmm.plan("grow", 4)
    record = vmm.StepRecord({})
    owner._register(0, steps[0], record)
    owner._register(2, steps[2], record)
    assert owner.agent.registered == [(1 << 40, 4 * G), ((1 << 40) + 2 * G, G)]
    full = _owner(_case(register="full"))
    full._register(2, steps[2], record)
    assert full.agent.registered == [(1 << 40, 4 * G)]
    assert len(record.timings["register"]) == 3


def test_publish_times_metadata_and_load(no_gpu):
    owner = _owner(_case(), {"load": {"load_ns": 7}})
    record = vmm.StepRecord({})
    owner._publish(record)
    owner._publish(record)
    assert record.timings["load"] == [7, 7] and len(record.timings["metadata"]) == 2
    assert owner.channel.sent[0]["blob"] == b"blob-0"
    assert record.info["blob_bytes"] == len(b"blob-0")


def test_transfer_records_every_call(no_gpu):
    owner = _owner(_case(), {"xfer": _done})
    owner.pattern = {1: vmm.NEW}
    record = vmm.StepRecord({})
    owner._transfer(1, record)
    for prefix in ("write", "read"):
        assert record.timings[f"{prefix}_prep"] == [1]
        assert record.timings[f"{prefix}_xfer"] == [2]
        assert record.timings[f"{prefix}_steady_xfer"] == [3, 4]
    assert [message["iters"] for message in owner.channel.sent] == [2, 2]
    assert not any(name == "memset" for name, _ in no_gpu)


def test_transfer_error_fails_the_run(no_gpu):
    owner = _owner(_case(), {"xfer": {"state": "ERR", "prep_ns": 1, "xfer_ns": 2}})
    owner.pattern = {1: vmm.NEW}
    with pytest.raises(vmm.VmmPerftestError, match="WRITE of chunk 1 ended in ERR"):
        owner._transfer(1, vmm.StepRecord({}))


def test_verify_fills_and_checks(no_gpu, monkeypatch):
    owner = _owner(_case(ops=("WRITE",)), {"xfer": _done}, verify=True)
    owner.pattern = {1: vmm.NEW}
    owner._transfer(1, vmm.StepRecord({}))
    assert ("memset", ((1 << 40) + G, 4096, vmm.NEW)) in no_gpu
    monkeypatch.setattr(vmm, "edges_match", lambda *args: False)
    with pytest.raises(vmm.VmmPerftestError, match="wrong bytes"):
        owner._transfer(1, vmm.StepRecord({}))

    reader = _owner(
        _case(ops=("READ",)), {"xfer": {**_done({}), "read_ok": False}}, verify=True
    )
    reader.pattern = {1: vmm.NEW}
    with pytest.raises(vmm.VmmPerftestError, match="READ of chunk 1: wrong bytes"):
        reader._transfer(1, vmm.StepRecord({}))


def test_span_read_pieces(no_gpu):
    owner = _owner(_case(ops=("READ",)), {"xfer": _done})
    owner.pattern = {0: vmm.OLD, 1: vmm.NEW, 2: vmm.OLD}
    record = vmm.StepRecord({})
    owner._span_read(0, 3, record)
    message = owner.channel.sent[0]
    assert message["address"] == 1 << 40 and message["length"] == 3 * G
    assert message["pieces"] == [[0, G, vmm.OLD], [G, G, vmm.NEW], [2 * G, G, vmm.OLD]]
    assert record.timings["span_read_xfer"] == [2]


def test_hole_transfers_are_timed_whatever_the_outcome(no_gpu):
    states = iter(
        [
            {"state": "ERR", "prep_ns": 5},
            {"state": "DONE", "prep_ns": 1, "xfer_ns": 2},
            {"state": "ERR", "prep_ns": 3, "xfer_ns": 4},
        ]
    )
    owner = _owner(_case(), {"xfer": lambda message: next(states)})
    owner.pattern = {0: vmm.OLD, 1: vmm.OLD, 2: vmm.OLD}
    record = vmm.StepRecord({})
    owner._hole(3, record)
    assert record.timings["hole_write_prep"] == [5]
    assert "hole_write_xfer" not in record.timings
    assert record.timings["hole_read_xfer"] == [2]
    assert record.timings["hole_span_read_xfer"] == [4]
    half = owner.channel.sent[2]
    assert half["address"] == (1 << 40) + 3 * G - G // 2 and half["length"] == G
    assert all(
        message["hole"] and not message["verify"] for message in owner.channel.sent
    )


def test_hole_transfer_that_never_finishes_fails_the_run(no_gpu):
    owner = _owner(_case(), {"xfer": {"state": "TIMEOUT", "prep_ns": 1}})
    owner.pattern = {0: vmm.OLD}
    with pytest.raises(vmm.VmmPerftestError, match="did not finish"):
        owner._hole(1, vmm.StepRecord({}))


def test_peer_error_names_the_step(no_gpu):
    class Refusing(FakeChannel):
        def request(self, message):
            raise vmm.VmmPerftestError("peer load: nixlNotAllowedError: x")

    owner = _owner(_case())
    owner.channel = Refusing()
    with pytest.raises(vmm.VmmPerftestError, match="^here: peer load"):
        owner._publish(vmm.StepRecord({}))


def test_memory_change(no_gpu, monkeypatch):
    owner = _owner(_case())
    values = iter([1000 - 3 * G, 1000 - 2 * G])
    monkeypatch.setattr(vmm, "free_bytes", lambda: next(values))
    owner.last_free = 1000
    first, second = vmm.StepRecord({}), vmm.StepRecord({})
    owner._memory(first)
    owner._memory(second)
    assert first.info["owner_free_change_bytes"] == -3 * G
    assert second.info["owner_free_change_bytes"] == G


# =============================================================================
# Peer, with the agent replaced
# =============================================================================


class FakePeerAgent:
    def __init__(self, prepare_raises=False, states=("DONE",)):
        self.prepare_raises = prepare_raises
        self.states = list(states)
        self.kept = []

    def xfer_descs(self, address, length, device):
        return (address, length, device)

    def prepare(self, op, local, remote):
        if self.prepare_raises:
            raise RuntimeError("nixlBackendError")
        return "handle"

    def post(self, handle):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def keep(self, handle):
        self.kept.append(handle)

    def release(self, handle):
        pass


def _peer(agent):
    peer = vmm.Peer(FakeChannel(), 1.0)
    peer.agent = agent
    peer.local = vmm.Arena(0, 1 << 41, G, G, "posix")
    return peer


def _xfer_message(**changes):
    base = dict(
        op="READ",
        address=1 << 40,
        length=4096,
        device=0,
        iters=3,
        hole=False,
        verify=False,
        pieces=[[0, 4096, vmm.OLD]],
    )
    return {**base, **changes}


def test_peer_refused_prepare(no_gpu):
    hole = _peer(FakePeerAgent(prepare_raises=True)).xfer(_xfer_message(hole=True))
    assert hole["state"] == "ERR" and "prep_ns" in hole and "xfer_ns" not in hole
    with pytest.raises(RuntimeError):
        _peer(FakePeerAgent(prepare_raises=True)).xfer(_xfer_message())


def test_peer_steady_and_timeout(no_gpu):
    reply = _peer(FakePeerAgent()).xfer(_xfer_message(verify=True))
    assert reply["state"] == "DONE" and len(reply["steady"]) == 3
    assert reply["read_ok"] is True
    assert "transport" not in reply
    agent = FakePeerAgent(states=("TIMEOUT",))
    reply = _peer(agent).xfer(_xfer_message())
    assert reply["state"] == "TIMEOUT" and reply["steady"] == [] and agent.kept


# =============================================================================
# Owner and peer together, two threads over an in-memory allgather
# =============================================================================


class ThreadDist:
    def __init__(self, rank, shared):
        self.rank = rank
        self.shared = shared

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return 2

    def allgather_obj(self, obj, timeout_sec=600):
        slots, barrier = self.shared
        slots[self.rank] = obj
        barrier.wait(timeout_sec)
        result = list(slots)
        barrier.wait(timeout_sec)
        return result


class LoopAgent:
    fail_load_at = None

    def __init__(self, name, device, timeout):
        self.stuck = []
        self.loads = 0

    def reg_descs(self, start, length):
        return (start, length)

    def register(self, descs):
        pass

    def partial_metadata(self, descs):
        return b"blob"

    def agent_metadata(self):
        return b"blob"

    def load(self, blob):
        self.loads += 1
        if self.loads == LoopAgent.fail_load_at:
            raise RuntimeError("nixlNotAllowedError: metadata update")

    def xfer_descs(self, address, length, device):
        return (address, length)

    def prepare(self, op, local, remote):
        return "handle"

    def post(self, handle):
        return "DONE"

    def release(self, handle):
        pass

    def close(self):
        pass


def _run_both(monkeypatch, config, json_path):
    monkeypatch.setattr(vmm, "Agent", LoopAgent)
    monkeypatch.setattr(vmm, "init_cuda", lambda device: None)
    monkeypatch.setattr(vmm, "granularity", lambda device: G)
    monkeypatch.setattr(
        vmm,
        "reserve",
        lambda size, device, gran, kind: vmm.Arena(0, 1 << 40, size, G, kind),
    )
    monkeypatch.setattr(vmm, "destroy", lambda arena: None)
    monkeypatch.setattr(vmm, "release_retired", lambda arena: None)
    shared = ([None, None], threading.Barrier(2))
    errors = {}

    def rank(index):
        try:
            perftest = vmm.VMMPerftest(config, dist=ThreadDist(index, shared))
            perftest.run(json_output_path=json_path if index == 0 else None)
        except Exception as exc:
            errors[index] = exc

    threads = [threading.Thread(target=rank, args=(index,)) for index in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not any(thread.is_alive() for thread in threads)
    return errors


def test_owner_and_peer_lockstep(no_gpu, monkeypatch, tmp_path):
    pytest.importorskip("tabulate")
    LoopAgent.fail_load_at = None
    config = vmm.VmmPerftestConfig.from_dict(
        {
            "path": "nvlink",
            "patterns": ["grow", "shrink"],
            "register": ["full"],
            "metadata": ["partial"],
            "chunk_units": [1],
            "chunks": 3,
            "iters": 2,
            "warmup_iters": 1,
            "xfer_iters": 1,
        }
    )
    json_path = tmp_path / "vmm.json"
    assert _run_both(monkeypatch, config, str(json_path)) == {}
    results = json.loads(json_path.read_text())
    assert "finished_ts" in results["metadata"]
    grow, shrink = results["scenarios"]
    assert [len(scenario["steps"]) for scenario in (grow, shrink)] == [3, 3]
    first = grow["steps"][0]["timings_ms"]
    assert {"reserve", "map", "register", "metadata", "load"} <= set(first)
    assert {"write_xfer", "read_xfer", "write_steady_xfer"} <= set(first)
    assert "hole_read_xfer" in shrink["steps"][1]["timings_ms"]
    assert "span_read_xfer" in grow["steps"][2]["timings_ms"]


def test_failed_call_fails_the_run_and_stops_the_peer(no_gpu, monkeypatch, tmp_path):
    pytest.importorskip("tabulate")
    LoopAgent.fail_load_at = 2
    config = vmm.VmmPerftestConfig.from_dict(
        {"path": "rdma", "patterns": ["grow"], "register": ["full"]}
    )
    try:
        errors = _run_both(monkeypatch, config, str(tmp_path / "vmm.json"))
    finally:
        LoopAgent.fail_load_at = None
    assert set(errors) == {0}
    assert "step 1" in str(errors[0]) and "nixlNotAllowedError" in str(errors[0])


# =============================================================================
# Report
# =============================================================================


def test_summarize_and_table():
    pytest.importorskip("tabulate")
    case = _case()
    record = vmm.StepRecord(
        {"step": 0, "action": "map", "changed": [0], "blob_bytes": 10}
    )
    samples = [{"register": [1_000_000, 3_000_000], "load": [2_000_000]}]
    scenario = vmm.summarize(case, [record], samples)
    timings = scenario["steps"][0]["timings_ms"]
    assert set(timings) == {"register", "load"} and timings["register"]["p50"] == 3.0
    assert scenario["case"]["chunk_bytes"] == G
    table = vmm.scenario_table(scenario)
    assert "Register" in table and "Load" in table and "Hole READ" not in table


# =============================================================================
# UCX_PROTO_INFO tables
# =============================================================================

_P = "[1792497793.700984] [lyris0125:2966908:0]   "
NVLINK_TABLES = [
    _P + "| remote memory write by ucp_put* from host memory to cuda/dev[5] |",
    _P + "| Range  |    Description     |              Config               |",
    _P + "| 0..inf | software emulation | tcp/lo                            |",
    _P + "| remote memory read by ucp_get*(multi) into cuda/GPU1 from cuda/dev[4] |",
    _P + "| Range  | Description |                     Config                     |",
    _P + "| 0..inf | zero-copy   | cuda_ipc/cuda                                  |",
    _P + "| remote memory write by ucp_put*(multi) from cuda/GPU1 to cuda/dev[4] |",
    _P + "| Range  | Description |                    Config                     |",
    _P + "| 0..inf | zero-copy   | cuda_ipc/cuda                                 |",
]
RDMA_TABLES = [
    _P
    + "| remote memory write by ucp_put*(fast-completion) from cuda/GPU0 to cuda/dev[5] |",
    _P
    + "| Range       | Description | Config                                         |",
    _P
    + "|       0..2K | short       | rc_mlx5/mlx5_0:1/path0                         |",
    _P
    + "| 2049..11598 | copy-in     | rc_mlx5/mlx5_0:1/path0                         |",
    _P
    + "|  11599..inf | zero-copy   | rc_mlx5/mlx5_0:1 50% on path0 and 50% on path1 |",
]
EMULATED_WRITE = [
    _P + "| remote memory write by ucp_put*(multi) from cuda/GPU1 to cuda |",
    _P + "| Range  |    Description     |              Config              |",
    _P + "| 0..inf | software emulation | rc_mlx5/mlx5_0:1/path0           |",
]


def test_transport_tables():
    assert vmm.transport(NVLINK_TABLES, "WRITE", 4096) == "zero-copy cuda_ipc/cuda"
    assert vmm.transport(NVLINK_TABLES, "READ", 4096) == "zero-copy cuda_ipc/cuda"
    assert vmm.transport(NVLINK_TABLES[:3], "WRITE", 4096) == ""
    assert vmm.transport(RDMA_TABLES, "WRITE", 1024) == "short rc_mlx5/mlx5_0:1/path0"
    assert vmm.transport(RDMA_TABLES, "WRITE", 4096) == "copy-in rc_mlx5/mlx5_0:1/path0"
    assert vmm.transport(RDMA_TABLES, "WRITE", 1 << 20).startswith("zero-copy rc_mlx5/")
    assert vmm.transport(RDMA_TABLES, "READ", 4096) == ""


def test_transport_per_transfer():
    class Lines:
        now: list = []

        def lines(self):
            return list(self.now)

    capture = Lines()
    tracker = vmm.TransportTracker(capture)
    capture.now = NVLINK_TABLES[6:9]
    assert tracker.after("WRITE", 4096, False, "DONE") == "zero-copy cuda_ipc/cuda"
    capture.now += EMULATED_WRITE
    assert tracker.after("WRITE", 4096, True, "ERR") == ""
    assert tracker.after("WRITE", 4096, False, "DONE") == "zero-copy cuda_ipc/cuda"
    capture.now += EMULATED_WRITE
    assert (
        tracker.after("WRITE", 4096, False, "DONE")
        == "software emulation rc_mlx5/mlx5_0:1/path0"
    )
    assert tracker.after("READ", 4096, False, "DONE") == "unknown"
    capture.now += NVLINK_TABLES[3:6]
    assert tracker.after("READ", 4096, False, "DONE") == "zero-copy cuda_ipc/cuda"


@pytest.mark.skipif(sys.platform != "linux", reason="tees file descriptors")
def test_capture_tees_and_keeps_tables():
    script = (
        "import os, sys\n"
        f"sys.path.insert(0, {os.path.join(p, '..')!r})\n"
        "import test.vmm_dynamic_buffer_perftest as vmm\n"
        "capture = vmm.Capture()\n"
        f"os.write(1, {NVLINK_TABLES[6]!r}.encode() + b'\\n')\n"
        f"print({NVLINK_TABLES[8]!r})\n"
        "lines = capture.lines()\n"
        "print('RESULT ' + vmm.transport(lines, 'WRITE', 4096), file=sys.stderr)\n"
        "capture.lines()\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert "RESULT zero-copy cuda_ipc/cuda" in done.stderr, done.stderr
    assert "cuda_ipc/cuda" in done.stdout
    assert "@@vmm-proto-sync" not in done.stdout + done.stderr
