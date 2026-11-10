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

"""VMM dynamic-buffer perftest: the time of every NIXL call while one buffer changes.

Rank 0 is the owner. It reserves one virtual-address range of `chunks` chunks and
changes its mapping one chunk per step, following a pattern. After every change it
calls register_memory again and builds new metadata. Rank 1 is the peer. It loads
that metadata and transfers to the chunk that changed. Each call is timed on its own:

    step 0:  reserve, map, register, metadata, load, WRITE, READ
    step 1:  map, register, metadata, load, WRITE, READ
    ...

Patterns:
    grow    start with chunk 0, map 1, 2, ... up to the last chunk
    shrink  start with every chunk, unmap the last one until only chunk 0 is left.
            Each unmap step also transfers into the chunk just unmapped.
    ends    start with the first and last chunk, map the middle from both ends inward
    remap   start with every chunk, replace each chunk's backing one per step

When READ runs, a step also READs the run of mapped chunks around the target in one
request. Every repeat uses fresh agents and a fresh buffer in the same processes, so
state that UCX caches per process (for example imported memory handles) can carry
over from one repeat to the next.

A NIXL call that raises, or a transfer that ends in error, fails the run. The one
exception is a transfer into a hole: it is timed until it completes or returns an
error, whichever happens.
"""

import ctypes
import itertools
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tabulate import tabulate

from nixl.logging import get_logger

logger = get_logger(__name__)

PATHS = ("nvlink", "rdma")
PATTERNS = ("grow", "shrink", "ends", "remap")
REGISTERS = ("full", "partial")
METADATA = ("partial", "agent")
OPS = ("WRITE", "READ")
HANDLE_TYPES = {"nvlink": "fabric", "rdma": "posix"}

OWNER_RANK = 0
PEER_RANK = 1
DEVICE = 0

OLD = 0x5A
NEW = 0xA5
WRITTEN = 0xC3

TIMINGS = [
    "reserve",
    "map",
    "register",
    "metadata",
    "load",
    "write_prep",
    "write_xfer",
    "write_steady_xfer",
    "read_prep",
    "read_xfer",
    "read_steady_xfer",
    "span_read_prep",
    "span_read_xfer",
    "span_read_steady_xfer",
    "hole_write_prep",
    "hole_write_xfer",
    "hole_read_prep",
    "hole_read_xfer",
    "hole_span_read_prep",
    "hole_span_read_xfer",
]

TABLE_COLUMNS = [
    ("map", "Map"),
    ("register", "Register"),
    ("metadata", "Metadata"),
    ("load", "Load"),
    ("write_prep", "WRITE prep"),
    ("write_xfer", "WRITE first"),
    ("write_steady_xfer", "WRITE steady"),
    ("read_prep", "READ prep"),
    ("read_xfer", "READ first"),
    ("read_steady_xfer", "READ steady"),
    ("span_read_xfer", "Span READ first"),
    ("hole_write_xfer", "Hole WRITE"),
    ("hole_read_xfer", "Hole READ"),
    ("hole_span_read_xfer", "Half-hole READ"),
]


class VmmPerftestError(RuntimeError):
    pass


# =============================================================================
# Patterns
# =============================================================================


@dataclass(frozen=True)
class Step:
    action: str  # map, remap, or unmap
    chunks: Tuple[int, ...]  # chunks this step changes
    target: int  # chunk the peer transfers to
    mapped: Tuple[int, ...]  # mapped chunks after the step
    hole: Optional[int] = None  # chunk just unmapped


def plan(pattern: str, count: int) -> List[Step]:
    if count < 2:
        raise ValueError("a pattern needs at least 2 chunks")
    first: Tuple[int, ...]
    changes: List[Tuple[str, int]]
    if pattern == "grow":
        first = (0,)
        changes = [("map", index) for index in range(1, count)]
    elif pattern == "shrink":
        first = tuple(range(count))
        changes = [("unmap", index) for index in range(count - 1, 0, -1)]
    elif pattern == "ends":
        first = (0, count - 1)
        changes = [("map", index) for index in _inward(count)]
    elif pattern == "remap":
        first = tuple(range(count))
        changes = [("remap", index) for index in range(count)]
    else:
        raise ValueError(f"unknown pattern {pattern}")

    mapped = set(first)
    steps = [Step("map", first, first[0], tuple(sorted(mapped)))]
    for action, chunk in changes:
        hole: Optional[int] = None
        if action == "unmap":
            mapped.discard(chunk)
            target, hole = max(mapped), chunk
        else:
            mapped.add(chunk)
            target = chunk
        steps.append(Step(action, (chunk,), target, tuple(sorted(mapped)), hole))
    return steps


def span(step: Step) -> Tuple[int, int]:
    """First chunk and chunk count of the mapped run that contains the target."""
    mapped = set(step.mapped)
    first = last = step.target
    while first - 1 in mapped:
        first -= 1
    while last + 1 in mapped:
        last += 1
    return first, last - first + 1


def _inward(count: int) -> List[int]:
    left, right, order = 1, count - 2, []
    while left <= right:
        order.append(left)
        if left != right:
            order.append(right)
        left, right = left + 1, right - 1
    return order


# =============================================================================
# Configuration
# =============================================================================


def _as_tuple(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class VmmPerftestConfig:
    path: str
    patterns: Tuple[str, ...] = PATTERNS
    register: Tuple[str, ...] = REGISTERS
    metadata: Tuple[str, ...] = METADATA
    ops: Tuple[str, ...] = OPS
    chunk_units: Tuple[int, ...] = (1, 4, 16)
    chunks: int = 8
    payload: int = 4096
    iters: int = 5
    warmup_iters: int = 1
    xfer_iters: int = 10
    transfer_timeout_sec: float = 15.0
    exchange_timeout_sec: float = 300.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VmmPerftestConfig":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        if "path" not in data:
            raise ValueError("Config must set 'path' to one of " + ", ".join(PATHS))
        values = dict(data)
        for key in ("patterns", "register", "metadata", "ops", "chunk_units"):
            if key in values:
                values[key] = _as_tuple(values[key])
        if "ops" in values:
            values["ops"] = tuple(
                dict.fromkeys(str(op).upper() for op in values["ops"])
            )
        if "chunk_units" in values:
            values["chunk_units"] = tuple(int(units) for units in values["chunk_units"])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        checks = [
            (self.path in PATHS, f"path must be one of {PATHS}"),
            (
                bool(self.patterns) and set(self.patterns) <= set(PATTERNS),
                f"patterns must be taken from {PATTERNS}",
            ),
            (
                bool(self.register) and set(self.register) <= set(REGISTERS),
                f"register must be taken from {REGISTERS}",
            ),
            (
                bool(self.metadata) and set(self.metadata) <= set(METADATA),
                f"metadata must be taken from {METADATA}",
            ),
            (
                bool(self.ops) and set(self.ops) <= set(OPS),
                f"ops must be taken from {OPS}",
            ),
            (
                bool(self.chunk_units) and min(self.chunk_units) >= 1,
                "chunk_units must be 1 or more",
            ),
            (self.chunks >= 2, "chunks must be 2 or more"),
            (self.payload >= 8, "payload must be 8 bytes or more"),
            (self.iters >= 1, "iters must be 1 or more"),
            (self.warmup_iters >= 0, "warmup_iters must not be negative"),
            (self.xfer_iters >= 0, "xfer_iters must not be negative"),
            (self.transfer_timeout_sec > 0, "transfer_timeout_sec must be positive"),
            (self.exchange_timeout_sec > 0, "exchange_timeout_sec must be positive"),
        ]
        for ok, reason in checks:
            if not ok:
                raise ValueError(reason)


@dataclass(frozen=True)
class Case:
    path: str
    pattern: str
    register: str
    metadata: str
    ops: Tuple[str, ...]
    chunk_units: int
    chunks: int
    granularity: int
    payload: int
    xfer_iters: int

    @property
    def chunk(self) -> int:
        return self.chunk_units * self.granularity

    @property
    def reserved(self) -> int:
        return self.chunks * self.chunk

    def name(self) -> str:
        return (
            f"pattern={self.pattern} register={self.register} "
            f"metadata={self.metadata} chunk_units={self.chunk_units} "
            f"chunks={self.chunks}"
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["ops"] = list(self.ops)
        data["chunk_bytes"] = self.chunk
        data["reserved_bytes"] = self.reserved
        return data


def cases(config: VmmPerftestConfig, granularity: int) -> List[Case]:
    found = [
        Case(
            config.path,
            pattern,
            register,
            metadata,
            config.ops,
            units,
            config.chunks,
            granularity,
            config.payload,
            config.xfer_iters,
        )
        for pattern, register, metadata, units in itertools.product(
            config.patterns, config.register, config.metadata, config.chunk_units
        )
    ]
    for case in found:
        if case.payload > case.chunk:
            raise ValueError(
                f"payload {case.payload} is larger than one chunk "
                f"({case.chunk} bytes at chunk_units={case.chunk_units})"
            )
    return found


# =============================================================================
# Timing and statistics
# =============================================================================


class Timer:
    ns = 0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.ns = time.perf_counter_ns() - self._start


def stats_ms(values_ns: List[int]) -> Optional[Dict[str, float]]:
    """avg, p50, p90, p99, min, max in milliseconds, as sequential-ct-perftest reports them."""
    if not values_ns:
        return None
    ordered = sorted(values_ns)
    n = len(ordered)
    return {
        "avg": sum(ordered) / n / 1e6,
        "p50": ordered[n // 2] / 1e6,
        "p90": ordered[int(n * 0.9)] / 1e6,
        "p99": (ordered[int(n * 0.99)] if n >= 100 else ordered[-1]) / 1e6,
        "min": ordered[0] / 1e6,
        "max": ordered[-1] / 1e6,
    }


# =============================================================================
# CUDA VMM
# =============================================================================


class CudaCallError(RuntimeError):
    pass


@dataclass
class Chunk:
    offset: int
    size: int
    handle: Any
    retired: bool = False


@dataclass
class Arena:
    device_id: int
    va_base: int
    reserved_size: int
    granularity: int
    handle_type: str
    chunks: List[Chunk] = field(default_factory=list)


def _driver():
    try:
        from cuda.bindings import driver as cuda
    except ImportError as exc:
        raise CudaCallError(
            "vmm-perftest needs cuda-python (pip install cuda-python)"
        ) from exc
    return cuda


def _check(cuda, result, function: str):
    err = result[0] if isinstance(result, tuple) else result
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise CudaCallError(f"{function}: {err}")
    if isinstance(result, tuple):
        return result[1] if len(result) == 2 else result[1:]
    return None


def init_cuda(device_id: int) -> None:
    cuda = _driver()
    _check(cuda, cuda.cuInit(0), "cuInit")
    device = _check(cuda, cuda.cuDeviceGet(device_id), "cuDeviceGet")
    context = _check(
        cuda, cuda.cuDevicePrimaryCtxRetain(device), "cuDevicePrimaryCtxRetain"
    )
    _check(cuda, cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent")


def granularity(device_id: int) -> int:
    cuda = _driver()
    flag = cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    value = _check(
        cuda,
        cuda.cuMemGetAllocationGranularity(_prop(cuda, device_id, "none"), flag),
        "cuMemGetAllocationGranularity",
    )
    return int(value)


def _prop(cuda, device_id: int, handle_type: str):
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    if hasattr(prop, "allocFlags"):
        prop.allocFlags.compressionType = 0
        if handle_type == "posix":
            prop.allocFlags.gpuDirectRDMACapable = 1
    handle_types = cuda.CUmemAllocationHandleType
    if handle_type == "fabric":
        prop.requestedHandleTypes = handle_types.CU_MEM_HANDLE_TYPE_FABRIC
    elif handle_type == "posix":
        prop.requestedHandleTypes = (
            handle_types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
    return prop


def reserve(size: int, device_id: int, gran: int, handle_type: str) -> Arena:
    cuda = _driver()
    va = _check(cuda, cuda.cuMemAddressReserve(size, gran, 0, 0), "cuMemAddressReserve")
    return Arena(device_id, int(va), size, gran, handle_type)


def map_chunk(arena: Arena, offset: int, size: int) -> None:
    cuda = _driver()
    prop = _prop(cuda, arena.device_id, arena.handle_type)
    handle = _check(cuda, cuda.cuMemCreate(size, prop, 0), "cuMemCreate")
    va = arena.va_base + offset
    try:
        _check(cuda, cuda.cuMemMap(va, size, 0, handle, 0), "cuMemMap")
        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = arena.device_id
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check(cuda, cuda.cuMemSetAccess(va, size, [access], 1), "cuMemSetAccess")
    except CudaCallError:
        cuda.cuMemRelease(handle)
        raise
    arena.chunks.append(Chunk(offset, size, handle))


def unmap_chunk(arena: Arena, offset: int, release: bool) -> Chunk:
    cuda = _driver()
    chunk = next(
        (item for item in arena.chunks if item.offset == offset and not item.retired),
        None,
    )
    if chunk is None:
        raise CudaCallError(f"cuMemUnmap: nothing mapped at offset {offset}")
    _check(
        cuda, cuda.cuMemUnmap(arena.va_base + chunk.offset, chunk.size), "cuMemUnmap"
    )
    if release:
        _check(cuda, cuda.cuMemRelease(chunk.handle), "cuMemRelease")
        arena.chunks.remove(chunk)
    else:
        chunk.retired = True
    return chunk


def remap_chunk(arena: Arena, offset: int) -> None:
    """Unmap the old allocation, keep it until release_retired, map a new one there."""
    old = unmap_chunk(arena, offset, release=False)
    map_chunk(arena, old.offset, old.size)


def release_retired(arena: Arena) -> None:
    cuda = _driver()
    for chunk in [item for item in arena.chunks if item.retired]:
        cuda.cuMemRelease(chunk.handle)
        arena.chunks.remove(chunk)


def destroy(arena: Arena) -> None:
    cuda = _driver()
    for chunk in list(arena.chunks):
        if not chunk.retired:
            cuda.cuMemUnmap(arena.va_base + chunk.offset, chunk.size)
        cuda.cuMemRelease(chunk.handle)
    arena.chunks.clear()
    cuda.cuMemAddressFree(arena.va_base, arena.reserved_size)


def memset(ptr: int, size: int, value: int) -> None:
    """Fill and wait: UCX copies on its own streams."""
    cuda = _driver()
    _check(
        cuda,
        cuda.cuMemsetD8(cuda.CUdeviceptr(ptr), value & 0xFF, size),
        "cuMemsetD8",
    )
    _check(cuda, cuda.cuCtxSynchronize(), "cuCtxSynchronize")


def free_bytes() -> int:
    cuda = _driver()
    free, _total = _check(cuda, cuda.cuMemGetInfo(), "cuMemGetInfo")
    return int(free)


def _read_bytes(ptr: int, size: int) -> bytes:
    cuda = _driver()
    host = (ctypes.c_ubyte * size)()
    _check(cuda, cuda.cuMemcpyDtoH(host, cuda.CUdeviceptr(ptr), size), "cuMemcpyDtoH")
    return bytes(host)


def edges_match(ptr: int, size: int, value: int) -> bool:
    """The first and last 8 bytes all equal `value`."""
    expected = bytes([value & 0xFF]) * 8
    return (
        _read_bytes(ptr, 8) == expected and _read_bytes(ptr + size - 8, 8) == expected
    )


# =============================================================================
# Transport detection (--report-transport)
# =============================================================================

_HEADER = re.compile(
    r"\|\s*remote memory (write|read) by ucp_\w+\*\S*\s+(.*?)\s*\|\s*$"
)
_ANY_HEADER = re.compile(r"\|\s*.+ by ucp_\w+\*")
_ROW = re.compile(
    r"\|\s*(\d+[KMGT]?)\.\.(inf|\d+[KMGT]?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*$"
)
# "cuda/GPU1", "cuda/dev[4]", or a bare "cuda" when the remote key names no device.
_CUDA = re.compile(r"\bcuda\b")
_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
_MARK = "@@vmm-proto-sync"


def transport(lines: List[str], op: str, size: int) -> str:
    """'<description> <config>' of the latest GPU-to-GPU UCX_PROTO_INFO table, or ''."""
    want = "write" if op == "WRITE" else "read"
    found = ""
    current = False
    for line in lines:
        header = _HEADER.search(line)
        if header:
            current = (
                header.group(1) == want and len(_CUDA.findall(header.group(2))) >= 2
            )
            continue
        if _ANY_HEADER.search(line):
            current = False
            continue
        row = _ROW.search(line)
        if current and row and _size(row.group(1)) <= size <= _size(row.group(2)):
            found = f"{row.group(3)} {row.group(4)}"
    return found


def _size(text: str) -> float:
    if text == "inf":
        return float("inf")
    suffix = text[-1] if text[-1] in "KMGT" else ""
    return int(text.rstrip("KMGT")) * _UNITS[suffix]


class Capture:
    """Tee of fd 1 and fd 2. lines() returns every table line printed so far."""

    def __init__(self) -> None:
        self._lines: List[str] = []
        self._marks = 0
        self._sync = 0
        self._cond = threading.Condition()
        for fd in (1, 2):
            read, write = os.pipe()
            original = os.dup(fd)
            os.dup2(write, fd)
            os.close(write)
            threading.Thread(
                target=self._pump, args=(read, original), daemon=True
            ).start()

    def lines(self, timeout: float = 5.0) -> List[str]:
        _flush_c_stdio()
        sys.stdout.flush()
        sys.stderr.flush()
        with self._cond:
            self._sync += 1
            target = self._sync * 2
        for fd in (1, 2):
            os.write(fd, f"{_MARK}\n".encode())
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._marks < target and time.monotonic() < deadline:
                self._cond.wait(deadline - time.monotonic())
            return list(self._lines)

    def _pump(self, read: int, original: int) -> None:
        pending = b""
        while True:
            chunk = os.read(read, 65536)
            if not chunk:
                break
            pending += chunk
            *complete, pending = pending.split(b"\n")
            echo, kept, marks = [], [], 0
            for raw in complete:
                text = raw.decode(errors="replace")
                if text.endswith(_MARK):
                    marks += 1
                    raw, text = raw[: -len(_MARK)], text[: -len(_MARK)]
                    if not raw:
                        continue
                echo.append(raw + b"\n")
                if "|" in text:
                    kept.append(text)
            if echo:
                os.write(original, b"".join(echo))
            with self._cond:
                self._lines.extend(kept)
                self._marks += marks
                self._cond.notify_all()
        if pending:
            os.write(original, pending)


def _flush_c_stdio() -> None:
    try:
        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError, TypeError):
        pass


class TransportTracker:
    """The UCX table each transfer selected. UCX prints a table once per selection,
    so a transfer that reused an earlier selection reads the last table found for its
    direction. Tables printed by hole transfers are skipped."""

    def __init__(self, capture) -> None:
        self.capture = capture
        self.seen = {"WRITE": 0, "READ": 0}
        self.used: Dict[str, List[str]] = {}

    def after(self, op: str, length: int, hole: bool, state: str) -> str:
        lines = self.capture.lines()
        if hole:
            self.seen = dict.fromkeys(self.seen, len(lines))
            return ""
        new = lines[self.seen[op] :]
        self.seen[op] = len(lines)
        if transport(new, op, length):
            self.used[op] = new
        if state != "DONE":
            return ""
        return transport(self.used.get(op, []), op, length) or "unknown"


# =============================================================================
# NIXL agent
# =============================================================================


class Agent:
    def __init__(self, name: str, device: int, transfer_timeout_sec: float):
        from nixl._api import nixl_agent

        self.agent = nixl_agent(name)
        if "UCX" not in self.agent.get_plugin_list():
            raise VmmPerftestError("UCX plugin is not available")
        self.device = device
        self.transfer_timeout_sec = transfer_timeout_sec
        self.regs: List[Any] = []
        self.stuck: List[Any] = []
        self.remote = ""

    def reg_descs(self, start: int, length: int):
        return self.agent.get_reg_descs([(start, length, self.device, "")], "VRAM")

    def register(self, descs) -> None:
        self.regs.append(self.agent.register_memory(descs))

    def partial_metadata(self, descs) -> bytes:
        return self.agent.get_partial_agent_metadata(descs, True)

    def agent_metadata(self) -> bytes:
        return self.agent.get_agent_metadata()

    def load(self, blob: bytes) -> None:
        name = self.agent.add_remote_agent(blob)
        self.remote = name.decode() if isinstance(name, bytes) else name

    def xfer_descs(self, address: int, length: int, device: int):
        return self.agent.get_xfer_descs([(address, length, device)], "VRAM")

    def prepare(self, op: str, local, remote):
        return self.agent.initialize_xfer(op, local, remote, self.remote, b"vmm")

    def post(self, handle) -> str:
        """DONE, ERR, TIMEOUT, or ERR:<exception> when NIXL raises."""
        try:
            state = self.agent.transfer(handle)
            deadline = time.monotonic() + self.transfer_timeout_sec
            while state not in ("DONE", "ERR") and time.monotonic() < deadline:
                state = self.agent.check_xfer_state(handle)
        except Exception as exc:
            return f"ERR:{type(exc).__name__}: {exc}"
        return state if state in ("DONE", "ERR") else "TIMEOUT"

    def keep(self, handle) -> None:
        # The request is still active: releasing it, also from the handle's
        # finalizer, would abort it while UCX still owns it.
        self.stuck.append(handle)

    def release(self, handle) -> None:
        try:
            self.agent.release_xfer_handle(handle)
        except Exception:
            pass

    def close(self) -> None:
        for handle in self.regs:
            try:
                self.agent.deregister_memory(handle)
            except Exception:
                pass
        self.regs.clear()
        if self.remote:
            try:
                self.agent.remove_remote_agent(self.remote)
            except Exception:
                pass
            self.remote = ""


# =============================================================================
# Owner <-> peer requests over the kvbench runtime
# =============================================================================


class Channel:
    """One request from the owner, one reply from the peer, each an allgather."""

    def __init__(self, dist, timeout_sec: float):
        self.dist = dist
        self.timeout_sec = timeout_sec
        self.broken = False

    def _gather(self, obj: Any) -> List[Any]:
        try:
            return self.dist.allgather_obj(obj, timeout_sec=self.timeout_sec)
        except TimeoutError:
            self.broken = True
            raise

    def request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        self._gather(message)
        reply = self._gather(None)[PEER_RANK]
        if reply.get("error"):
            raise VmmPerftestError(f"peer {message['cmd']}: {reply['error']}")
        return reply

    def receive(self) -> Dict[str, Any]:
        return self._gather(None)[OWNER_RANK]

    def reply(self, reply: Dict[str, Any]) -> None:
        self._gather(reply)


# =============================================================================
# Owner: one pass of one case
# =============================================================================


@dataclass
class StepRecord:
    info: Dict[str, Any]
    timings: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))


def step_info(case: Case, index: int, step: Step) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "step": index,
        "action": step.action,
        "changed": list(step.chunks),
        "target": step.target,
        "mapped_chunks": len(step.mapped),
    }
    first, count = span(step)
    if "READ" in case.ops and count > 1:
        info["span_first"], info["span_chunks"] = first, count
    if step.hole is not None:
        info["hole_chunk"] = step.hole
    return info


@contextmanager
def _call(where: str, name: str) -> Iterator[None]:
    try:
        yield
    except VmmPerftestError:
        raise
    except Exception as exc:
        raise VmmPerftestError(f"{where}: {name}: {type(exc).__name__}: {exc}") from exc


class OwnerPass:
    def __init__(
        self,
        case: Case,
        channel: Channel,
        seq: int,
        verify: bool,
        report_transport: bool,
        transfer_timeout_sec: float,
    ):
        self.case = case
        self.channel = channel
        self.seq = seq
        self.verify = verify
        self.report_transport = report_transport
        self.transfer_timeout_sec = transfer_timeout_sec
        self.arena: Optional[Arena] = None
        self.agent: Optional[Agent] = None
        self.peer_open = False
        self.pattern: Dict[int, int] = {}
        self.last_blob = b""
        self.last_free: Optional[int] = None
        self.where = ""

    def run(self) -> List[StepRecord]:
        records = []
        try:
            for index, step in enumerate(plan(self.case.pattern, self.case.chunks)):
                self.where = f"{self.case.name()} step {index} ({step.action} {list(step.chunks)})"
                record = StepRecord(step_info(self.case, index, step))
                if index == 0:
                    self._open(record)
                self._change(index, step, record)
                self._register(index, step, record)
                self._publish(record)
                self._transfer(step.target, record)
                if "span_chunks" in record.info:
                    self._span_read(
                        record.info["span_first"], record.info["span_chunks"], record
                    )
                if step.hole is not None:
                    self._hole(step.hole, record)
                with _call(self.where, "release retired allocations"):
                    release_retired(self._arena())
                self._memory(record)
                records.append(record)
        finally:
            self._close()
        return records

    def _arena(self) -> Arena:
        assert self.arena is not None
        return self.arena

    def _agent(self) -> Agent:
        assert self.agent is not None
        return self.agent

    def _va(self, chunk: int) -> int:
        return self._arena().va_base + chunk * self.case.chunk

    def _peer(self, message: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self.channel.request(message)
        except VmmPerftestError as exc:
            raise VmmPerftestError(f"{self.where}: {exc}") from exc

    def _open(self, record: StepRecord) -> None:
        case = self.case
        with _call(self.where, "cuMemAddressReserve"):
            self.last_free = self._free()
            with Timer() as timer:
                self.arena = reserve(
                    case.reserved, DEVICE, case.granularity, HANDLE_TYPES[case.path]
                )
        record.timings["reserve"].append(timer.ns)
        with _call(self.where, "create agent"):
            self.agent = Agent(
                f"vmm-owner-{self.seq}", DEVICE, self.transfer_timeout_sec
            )
        self._peer(
            {
                "cmd": "open",
                "name": f"vmm-peer-{self.seq}",
                "path": case.path,
                "size": max(case.payload, case.reserved),
                "report_transport": self.report_transport,
            }
        )
        self.peer_open = True

    def _close(self) -> None:
        if self.peer_open and not self.channel.broken:
            try:
                self.channel.request({"cmd": "close"})
            except Exception:
                logger.exception("Peer close failed")
        if self.agent is not None:
            self.agent.close()
            self.agent = None
        if self.arena is not None:
            try:
                destroy(self.arena)
            except Exception:
                logger.exception("Releasing the owner buffer failed")
            self.arena = None

    @staticmethod
    def _free() -> Optional[int]:
        try:
            return free_bytes()
        except Exception:
            return None

    def _memory(self, record: StepRecord) -> None:
        free = self._free()
        if free is None:
            return
        if self.last_free is not None:
            record.info["owner_free_change_bytes"] = free - self.last_free
        self.last_free = free

    def _change(self, index: int, step: Step, record: StepRecord) -> None:
        chunk = self.case.chunk
        arena = self._arena()
        with _call(self.where, step.action):
            with Timer() as timer:
                for item in step.chunks:
                    if step.action == "map":
                        map_chunk(arena, item * chunk, chunk)
                    elif step.action == "remap":
                        remap_chunk(arena, item * chunk)
                    else:
                        unmap_chunk(arena, item * chunk, release=True)
            for item in step.chunks:
                if step.action == "unmap":
                    self.pattern.pop(item, None)
                    continue
                self.pattern[item] = OLD if index == 0 else NEW
                if self.verify:
                    memset(self._va(item), chunk, self.pattern[item])
        record.timings["map"].append(timer.ns)

    def _register(self, index: int, step: Step, record: StepRecord) -> None:
        arena, agent = self._arena(), self._agent()
        if index == 0 or self.case.register == "full":
            start, length = arena.va_base, arena.reserved_size
        else:
            start, length = self._va(step.chunks[0]), len(step.chunks) * self.case.chunk
        with _call(self.where, "register_memory"):
            descs = agent.reg_descs(start, length)
            with Timer() as timer:
                agent.register(descs)
        record.timings["register"].append(timer.ns)

    def _publish(self, record: StepRecord) -> None:
        arena, agent = self._arena(), self._agent()
        partial = self.case.metadata == "partial"
        name = "get_partial_agent_metadata" if partial else "get_agent_metadata"
        with _call(self.where, name):
            descs = agent.reg_descs(arena.va_base, arena.reserved_size)
            with Timer() as timer:
                if partial:
                    blob = bytes(agent.partial_metadata(descs))
                else:
                    blob = bytes(agent.agent_metadata())
        record.timings["metadata"].append(timer.ns)
        record.info["blob_bytes"] = len(blob)
        record.info["blob_changed"] = blob != self.last_blob
        self.last_blob = blob
        reply = self._peer({"cmd": "load", "blob": blob})
        record.timings["load"].append(reply["load_ns"])

    def _xfer(self, op: str, address: int, length: int, **extra) -> Dict[str, Any]:
        return self._peer(
            {
                "cmd": "xfer",
                "op": op,
                "address": address,
                "length": length,
                "device": DEVICE,
                "iters": 0,
                "hole": False,
                "verify": self.verify,
                **extra,
            }
        )

    def _record_xfer(
        self, prefix: str, reply: Dict[str, Any], record: StepRecord
    ) -> None:
        if "prep_ns" in reply:
            record.timings[f"{prefix}_prep"].append(reply["prep_ns"])
        if "xfer_ns" in reply:
            record.timings[f"{prefix}_xfer"].append(reply["xfer_ns"])
        if reply.get("steady"):
            record.timings[f"{prefix}_steady_xfer"].extend(reply["steady"])
        if reply.get("transport"):
            record.info.setdefault("transport", {})[prefix] = reply["transport"]

    def _transfer(self, target: int, record: StepRecord) -> None:
        """WRITE and READ of the target chunk, on the registration just published."""
        case = self.case
        for op in case.ops:
            if self.verify:
                with _call(self.where, "fill target"):
                    memset(self._va(target), case.payload, self.pattern[target])
            reply = self._xfer(
                op,
                self._va(target),
                case.payload,
                iters=case.xfer_iters,
                pieces=[[0, case.payload, self.pattern[target]]],
            )
            self._record_xfer(op.lower(), reply, record)
            if reply["state"] != "DONE":
                raise VmmPerftestError(
                    f"{self.where}: {op} of chunk {target} ended in {reply['state']}"
                )
            if not self.verify:
                continue
            if op == "WRITE":
                with _call(self.where, "verify WRITE"):
                    ok = edges_match(self._va(target), case.payload, WRITTEN)
            else:
                ok = bool(reply.get("read_ok"))
            if not ok:
                raise VmmPerftestError(
                    f"{self.where}: {op} of chunk {target}: wrong bytes"
                )

    def _span_read(self, first: int, count: int, record: StepRecord) -> None:
        """One READ over chunks first .. first+count-1. Each chunk keeps its pattern."""
        chunk = self.case.chunk
        pieces = [
            [index * chunk, chunk, self.pattern[first + index]]
            for index in range(count)
        ]
        if self.verify:
            with _call(self.where, "fill span"):
                for offset, length, value in pieces:
                    memset(self._va(first) + offset, length, value)
        reply = self._xfer(
            "READ",
            self._va(first),
            count * chunk,
            iters=self.case.xfer_iters,
            pieces=pieces,
        )
        self._record_xfer("span_read", reply, record)
        if reply["state"] != "DONE":
            raise VmmPerftestError(
                f"{self.where}: READ of chunks {first}..{first + count - 1} "
                f"ended in {reply['state']}"
            )
        if self.verify and not reply.get("read_ok"):
            raise VmmPerftestError(
                f"{self.where}: READ of chunks {first}..{first + count - 1}: "
                "wrong bytes"
            )

    def _hole(self, hole: int, record: StepRecord) -> None:
        """Transfers into the chunk just unmapped, then a READ from the middle of the
        chunk before it to the middle of the hole. Timed until they complete or fail."""
        transfers = [
            (f"hole_{op.lower()}", op, self._va(hole), self.case.payload)
            for op in self.case.ops
        ]
        if "READ" in self.case.ops and hole - 1 in self.pattern:
            chunk = self.case.chunk
            transfers.append(
                ("hole_span_read", "READ", self._va(hole) - chunk // 2, chunk)
            )
        for prefix, op, address, length in transfers:
            reply = self._xfer(op, address, length, hole=True, verify=False)
            if reply.get("state") == "TIMEOUT":
                raise VmmPerftestError(
                    f"{self.where}: {prefix} did not finish within "
                    f"{self.transfer_timeout_sec}s"
                )
            self._record_xfer(prefix, reply, record)


# =============================================================================
# Peer
# =============================================================================


class Peer:
    def __init__(self, channel: Channel, transfer_timeout_sec: float):
        self.channel = channel
        self.transfer_timeout_sec = transfer_timeout_sec
        self.agent: Optional[Agent] = None
        self.local: Optional[Arena] = None
        self.tracker: Optional[TransportTracker] = None
        self.capture: Optional[Capture] = None
        self.leaked: List[Any] = []

    def serve(self) -> None:
        handlers = {"open": self.open, "load": self.load, "xfer": self.xfer}
        while True:
            message = self.channel.receive()
            command = message.get("cmd")
            if command == "quit":
                self.close()
                self.channel.reply({})
                break
            try:
                if command == "close":
                    self.close()
                    reply: Dict[str, Any] = {}
                elif command in handlers:
                    reply = handlers[command](message)
                else:
                    reply = {"error": f"unknown command {command}"}
            except Exception as exc:
                reply = {"error": f"{type(exc).__name__}: {exc}"}
            self.channel.reply(reply)
        if self.leaked:
            # A normal exit runs the handle finalizers, which abort requests that UCX
            # still owns.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    def open(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """A fresh agent and one registered local buffer of `size` bytes."""
        self.close()
        if message["report_transport"] and self.capture is None:
            os.environ.setdefault("UCX_PROTO_INFO", "y")
            self.capture = Capture()
        self.tracker = TransportTracker(self.capture) if self.capture else None
        init_cuda(DEVICE)
        gran = granularity(DEVICE)
        size = -(-int(message["size"]) // gran) * gran
        self.local = reserve(size, DEVICE, gran, HANDLE_TYPES[message["path"]])
        map_chunk(self.local, 0, size)
        self.agent = Agent(message["name"], DEVICE, self.transfer_timeout_sec)
        self.agent.register(self.agent.reg_descs(self.local.va_base, size))
        return {}

    def load(self, message: Dict[str, Any]) -> Dict[str, Any]:
        assert self.agent is not None
        with Timer() as timer:
            self.agent.load(message["blob"])
        return {"load_ns": timer.ns}

    def xfer(self, message: Dict[str, Any]) -> Dict[str, Any]:
        assert self.agent is not None and self.local is not None
        op, length, hole = message["op"], int(message["length"]), message["hole"]
        verify = message["verify"]
        local = self.local.va_base
        if verify:
            memset(local, length, WRITTEN if op == "WRITE" else 0)
        local_descs = self.agent.xfer_descs(local, length, DEVICE)
        remote_descs = self.agent.xfer_descs(
            int(message["address"]), length, int(message["device"])
        )
        prep = Timer()
        try:
            with prep:
                handle = self.agent.prepare(op, local_descs, remote_descs)
        except Exception:
            if not hole:
                raise
            return {"state": "ERR", "prep_ns": prep.ns}
        with Timer() as first:
            state = self.agent.post(handle)
        steady = []
        for _ in range(int(message["iters"])):
            if state != "DONE":
                break
            with Timer() as timer:
                state = self.agent.post(handle)
            if state == "DONE":
                steady.append(timer.ns)
        if state == "TIMEOUT":
            self.agent.keep(handle)
        else:
            self.agent.release(handle)
        reply: Dict[str, Any] = {
            "state": state,
            "prep_ns": prep.ns,
            "xfer_ns": first.ns,
            "steady": steady,
        }
        if verify and op == "READ" and state == "DONE":
            reply["read_ok"] = all(
                edges_match(local + int(offset), int(size), int(value))
                for offset, size, value in message["pieces"]
            )
        if self.tracker is not None:
            reply["transport"] = self.tracker.after(op, length, hole, state)
        return reply

    def close(self) -> None:
        if self.agent is not None and self.agent.stuck:
            # The active request may still touch the registration and the buffer.
            self.leaked.append((self.agent, self.local))
            self.agent = self.local = None
            return
        if self.agent is not None:
            self.agent.close()
            self.agent = None
        if self.local is not None:
            try:
                destroy(self.local)
            except Exception:
                logger.exception("Releasing the peer buffer failed")
            self.local = None


# =============================================================================
# Report
# =============================================================================


def summarize(
    case: Case, records: List[StepRecord], samples: List[Dict[str, List[int]]]
) -> Dict[str, Any]:
    steps = []
    for record, step_samples in zip(records, samples):
        timings = {
            name: stats_ms(step_samples[name])
            for name in TIMINGS
            if step_samples.get(name)
        }
        steps.append({**record.info, "timings_ms": timings})
    return {"case": case.to_dict(), "steps": steps}


def scenario_table(scenario: Dict[str, Any]) -> str:
    steps = scenario["steps"]
    columns = [
        (name, label)
        for name, label in TABLE_COLUMNS
        if any(name in step["timings_ms"] for step in steps)
    ]
    headers = ["Step", "Action", "Chunks", "Blob (B)"] + [label for _, label in columns]
    rows = []
    for step in steps:
        timings = step["timings_ms"]
        rows.append(
            [
                step["step"],
                step["action"],
                ",".join(str(chunk) for chunk in step["changed"]),
                step.get("blob_bytes"),
            ]
            + [timings[name]["p50"] if name in timings else None for name, _ in columns]
        )
    return tabulate(rows, headers=headers, floatfmt=".3f", missingval="-")


# =============================================================================
# Entry point
# =============================================================================


class VMMPerftest:
    def __init__(
        self,
        config: VmmPerftestConfig,
        verify_buffers: bool = False,
        report_transport: bool = False,
        dist=None,
    ):
        if dist is None:
            from runtime.etcd_rt import etcd_dist_utils

            dist = etcd_dist_utils
        self.dist = dist
        self.config = config
        self.verify_buffers = verify_buffers
        self.report_transport = report_transport
        self.rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise ValueError(
                f"vmm-perftest runs on exactly 2 ranks (owner and peer), got {world_size}"
            )
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            logger.warning(
                "CUDA_VISIBLE_DEVICES is not set: each rank uses device 0 of what it "
                "sees, so both ranks may share one GPU"
            )

    def run(self, json_output_path: Optional[str] = None) -> None:
        channel = Channel(self.dist, self.config.exchange_timeout_sec)
        if self.rank == PEER_RANK:
            Peer(channel, self.config.transfer_timeout_sec).serve()
            return
        try:
            self._run_owner(channel, json_output_path)
        finally:
            if not channel.broken:
                channel.request({"cmd": "quit"})

    def _run_owner(self, channel: Channel, json_output_path: Optional[str]) -> None:
        config = self.config
        init_cuda(DEVICE)
        gran = granularity(DEVICE)
        results: Dict[str, Any] = {
            "metadata": {
                "started_ts": time.time(),
                "config": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in asdict(config).items()
                },
                "ucx_tls": os.environ.get("UCX_TLS", ""),
                "granularity_bytes": gran,
                "verify_buffers": self.verify_buffers,
                "report_transport": self.report_transport,
                "time_unit": "ms",
            },
            "scenarios": [],
        }
        logger.info(
            "VMM perftest: path=%s, UCX_TLS=%s, granularity=%d",
            config.path,
            results["metadata"]["ucx_tls"] or "<unset>",
            gran,
        )
        seq = 0
        for case in cases(config, gran):
            steps = plan(case.pattern, case.chunks)
            samples: List[Dict[str, List[int]]] = [defaultdict(list) for _ in steps]
            records: List[StepRecord] = []
            for iteration in range(config.warmup_iters + config.iters):
                records = OwnerPass(
                    case,
                    channel,
                    seq,
                    self.verify_buffers,
                    self.report_transport,
                    config.transfer_timeout_sec,
                ).run()
                seq += 1
                if iteration < config.warmup_iters:
                    continue
                for index, record in enumerate(records):
                    for name, values in record.timings.items():
                        samples[index][name].extend(values)
            scenario = summarize(case, records, samples)
            results["scenarios"].append(scenario)
            logger.info(
                "%s, %d iters (p50 ms)\n%s",
                case.name(),
                config.iters,
                scenario_table(scenario),
            )
            _save(results, json_output_path)
        results["metadata"]["finished_ts"] = time.time()
        _save(results, json_output_path)


def _save(results: Dict[str, Any], json_output_path: Optional[str]) -> None:
    """Written after every scenario, so a run that crashes keeps the finished ones."""
    if not json_output_path:
        return
    staged = f"{json_output_path}.tmp"
    with open(staged, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(staged, json_output_path)
