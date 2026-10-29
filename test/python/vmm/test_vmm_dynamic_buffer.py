# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic buffer semantics for CUDA VMM memory.

The owner reserves a virtual address range of several chunks, maps GPU memory into
some of them, and registers the whole range. A peer loads the owner's metadata and
reads or writes the owner's chunks while the owner maps, remaps, and unmaps them.

- The registration key is the full reserved range. registerMem with an exact match
  updates it in place. A descriptor fully inside a registration updates it; one
  that extends outside it is rejected.
- The metadata is stable across map, unmap, and remap. A peer follows every change
  without another loadRemote, and another loadRemote is a no-op that keeps the
  remote name.
- Access to a hole (an unmapped chunk) returns a NIXL error, and the peer stays
  usable afterwards. A timeout or a completed transfer fails.

Mapped memory is filled with 0x5A, a remap fills the new allocation with 0xA5 (or
0x3C), and writes use 0xC3. After a remap, the retired allocation must keep its
bytes.

The owner (GPU 0) and every peer (GPU 1) are separate processes, one agent each.
They take turns: between two syncs either the owner changes memory or the peers
transfer. At each sync the owner checks that the peers' writes landed and refills
the chunks they wrote. Every check is collected and reported at the end.

After each change the peers check the arena's layout: every chunk is read on its
own, every hole must return a NIXL error, the longest run of mapped chunks is
read and written as one request, and a request half in each hole and half in a
neighbouring mapped chunk must fail.

The tests need two GPUs, cuda-python (pip install cuda-python), and the UCX
backend. They are skipped unless NIXL_VMM_TEST names a path, e.g.:

    NIXL_VMM_TEST=nvlink pytest -s test/python/vmm
    NIXL_VMM_TEST=rdma pytest -s test/python/vmm -k shrink

NIXL_VMM_HANDLE (fabric or posix) overrides the allocation handle type, which
defaults to fabric for nvlink and posix for rdma. NIXL_VMM_CHUNK_UNITS sets the
chunk size in multiples of the allocation granularity (default 1). UCX_TLS and,
for nvlink, UCX_NET_DEVICES and UCX_RNDV_THRESH get defaults unless they are
already set.
"""

import json
import os
import sys
import tempfile
import time

import pytest
import torch

from nixl._api import nixl_agent, nixl_agent_config

OLD = 0x5A
NEW = 0xA5
REMAPPED = 0x3C
WRITE = 0xC3
PATTERNS = (OLD, NEW, REMAPPED, WRITE, 0)
PAYLOAD = 4096
LOCAL_CHUNKS = 4
# A span starts halfway into its first chunk and ends halfway into its last one.
MAX_SPAN_CHUNKS = LOCAL_CHUNKS + 1
OWNER_GPU = 0
PEER_GPU = 1
XFER_TIMEOUT_S = 10.0
SYNC_TIMEOUT_S = 300.0

UCX_ENV = {
    "rdma": {"UCX_TLS": "rc_x,ud_x,cuda_copy,tcp"},
    "nvlink": {
        "UCX_TLS": "self,sm,tcp,cuda_ipc,cuda_copy",
        "UCX_NET_DEVICES": "lo",
        "UCX_RNDV_THRESH": "0",
    },
}
DEFAULT_HANDLE = {"rdma": "posix", "nvlink": "fabric"}


@pytest.fixture
def vmm_config(backend_name, monkeypatch):
    path = os.environ.get("NIXL_VMM_TEST")
    if not path:
        pytest.skip("set NIXL_VMM_TEST=rdma or nvlink to run the VMM tests")
    if path not in UCX_ENV:
        pytest.fail(f"NIXL_VMM_TEST must be rdma or nvlink, not {path!r}")
    handle = os.environ.get("NIXL_VMM_HANDLE") or DEFAULT_HANDLE[path]
    if handle not in ("fabric", "posix"):
        pytest.fail(f"NIXL_VMM_HANDLE must be fabric or posix, not {handle!r}")
    try:
        import cuda.bindings.driver  # noqa: F401
    except ImportError:
        pytest.fail("NIXL_VMM_TEST needs cuda-python: pip install cuda-python")
    if torch.cuda.device_count() < 2:
        pytest.skip("the VMM tests need two GPUs: owner on GPU 0, peers on GPU 1")
    for key, value in UCX_ENV[path].items():
        if key not in os.environ:
            monkeypatch.setenv(key, value)
    return {
        "backend": backend_name,
        "handle": handle,
        "chunk_units": int(os.environ.get("NIXL_VMM_CHUNK_UNITS", "1")),
    }


# ----- CUDA VMM on the owner -----


def _driver():
    from cuda.bindings import driver

    return driver


def _cu(result, what):
    """Return the outputs of a cuda.bindings driver call; raise on an error."""
    if not isinstance(result, tuple):
        result = (result,)
    if result[0] != _driver().CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what}: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


class _DevicePtr:
    """Raw device memory as a torch tensor, through __cuda_array_interface__."""

    def __init__(self, ptr, size):
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _view(ptr, size, gpu):
    return torch.as_tensor(_DevicePtr(ptr, size), device=f"cuda:{gpu}")


class _NotAdjacent(RuntimeError):
    pass


class _Chunk:
    def __init__(self, offset, size, handle, owned=True):
        self.offset = offset
        self.size = size
        self.handle = handle
        # An alias of another arena's allocation is not released by this arena.
        self.owned = owned
        self.retired = False


class _Arena:
    def __init__(self, va, size):
        self.va = va
        self.size = size
        self.chunks = []

    def live(self, offset):
        for chunk in self.chunks:
            if chunk.offset == offset and not chunk.retired:
                return chunk
        raise ValueError(f"no mapped chunk at offset {offset:#x}")


class _Vmm:
    """Reserve, map, remap, alias, and unmap chunks of reserved ranges."""

    def __init__(self, gpu, handle_type, chunk_units):
        cuda = _driver()
        _cu(cuda.cuInit(0), "cuInit")
        self.gpu = gpu
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = gpu
        kinds = cuda.CUmemAllocationHandleType
        if handle_type == "fabric":
            prop.requestedHandleTypes = kinds.CU_MEM_HANDLE_TYPE_FABRIC
        else:
            prop.requestedHandleTypes = kinds.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            prop.allocFlags.gpuDirectRDMACapable = 1
        self.prop = prop
        flag = (
            cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        )
        result = cuda.cuMemGetAllocationGranularity(prop, flag)
        self.granularity = int(_cu(result, "cuMemGetAllocationGranularity"))
        self.chunk = self.granularity * chunk_units
        self.arenas = {}

    def reserve(self, name, chunks, after=None):
        """Reserve `chunks` chunks; with `after`, right where that arena ends."""
        cuda = _driver()
        addr = 0
        if after is not None:
            prev = self.arenas[after]
            addr = prev.va + prev.size
        size = chunks * self.chunk
        result = cuda.cuMemAddressReserve(size, self.granularity, addr, 0)
        va = int(_cu(result, "cuMemAddressReserve"))
        if addr and va != addr:
            cuda.cuMemAddressFree(va, size)
            raise _NotAdjacent(f"CUDA reserved {va:#x}, not {addr:#x}")
        self.arenas[name] = _Arena(va, size)

    def _access(self, va, size):
        cuda = _driver()
        desc = cuda.CUmemAccessDesc()
        desc.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = self.gpu
        desc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _cu(cuda.cuMemSetAccess(va, size, [desc], 1), "cuMemSetAccess")

    def map(self, name, index, pattern, count=1):
        """Map one new allocation over `count` chunks and fill it."""
        cuda = _driver()
        arena = self.arenas[name]
        offset, size = index * self.chunk, count * self.chunk
        handle = _cu(cuda.cuMemCreate(size, self.prop, 0), "cuMemCreate")
        _cu(cuda.cuMemMap(arena.va + offset, size, 0, handle, 0), "cuMemMap")
        self._access(arena.va + offset, size)
        arena.chunks.append(_Chunk(offset, size, handle))
        self.fill(name, index, pattern, count)

    def remap(self, name, index, pattern):
        """Unmap a chunk, keep its allocation alive, and map a new one there."""
        arena = self.arenas[name]
        old = arena.live(index * self.chunk)
        _cu(_driver().cuMemUnmap(arena.va + old.offset, old.size), "cuMemUnmap")
        old.retired = True
        # retired_matches reads the most recently retired allocation, the last one.
        arena.chunks.remove(old)
        arena.chunks.append(old)
        self.map(name, index, pattern, old.size // self.chunk)

    def unmap(self, name, index):
        cuda = _driver()
        arena = self.arenas[name]
        chunk = arena.live(index * self.chunk)
        _cu(cuda.cuMemUnmap(arena.va + chunk.offset, chunk.size), "cuMemUnmap")
        if chunk.owned:
            _cu(cuda.cuMemRelease(chunk.handle), "cuMemRelease")
        arena.chunks.remove(chunk)

    def alias(self, name, index, source, source_index):
        """Map the allocation behind chunk `source_index` of `source` here too."""
        cuda = _driver()
        arena = self.arenas[name]
        src = self.arenas[source].live(source_index * self.chunk)
        va = arena.va + index * self.chunk
        _cu(cuda.cuMemMap(va, src.size, 0, src.handle, 0), "cuMemMap")
        self._access(va, src.size)
        arena.chunks.append(_Chunk(index * self.chunk, src.size, src.handle, False))

    def fill(self, name, index, pattern, count=1):
        va = self.arenas[name].va + index * self.chunk
        _view(va, count * self.chunk, self.gpu).fill_(pattern)
        torch.cuda.synchronize()

    def matches(self, ptr, pattern, size=8):
        return bool((_view(ptr, size, self.gpu) == pattern).all())

    def retired_matches(self, name, pattern):
        """Map the most recently retired allocation at a scratch address and check it."""
        cuda = _driver()
        chunk = [c for c in self.arenas[name].chunks if c.retired][-1]
        result = cuda.cuMemAddressReserve(chunk.size, self.granularity, 0, 0)
        scratch = int(_cu(result, "cuMemAddressReserve"))
        try:
            _cu(cuda.cuMemMap(scratch, chunk.size, 0, chunk.handle, 0), "cuMemMap")
            self._access(scratch, chunk.size)
            return self.matches(scratch, pattern, chunk.size)
        finally:
            cuda.cuMemUnmap(scratch, chunk.size)
            cuda.cuMemAddressFree(scratch, chunk.size)


# ----- one process: the owner or a peer -----


def _hex(pattern):
    return f"0x{pattern:02X}" if isinstance(pattern, int) else str(pattern)


def _failed_cleanly(state):
    """A completed NIXL error. A timeout or a completed transfer is not."""
    return state == "ERR" or state.startswith(("ERR:", "PREP_ERR:"))


def _span_points(chunk, length, patterns):
    """First and last 8 bytes of every chunk piece of a span, by offset in it."""
    points = {}
    edge = 0
    for index, pattern in enumerate(patterns):
        end = min(chunk // 2 + index * chunk, length)
        points[edge] = pattern
        points[end - 8] = pattern
        edge = end
    return points


def _longest_run(layout):
    """The longest run of consecutive mapped chunks, at most MAX_SPAN_CHUNKS long."""
    best, run = [], []
    for index in sorted(layout):
        if layout[index] is None or (run and index != run[-1] + 1):
            run = []
        if layout[index] is not None:
            run.append(index)
            if len(run) > len(best):
                best = list(run)
    return best[:MAX_SPAN_CHUNKS]


class _Rank:
    """The owner (rank 0) or one peer. Every scenario runs on all ranks."""

    def __init__(self, rank, names, exch, cfg):
        self.rank = rank
        self.names = names
        self.name = names[rank]
        self.owner = rank == 0
        self.peer = not self.owner
        self.exch = exch
        self.gpu = OWNER_GPU if self.owner else PEER_GPU
        torch.cuda.set_device(self.gpu)
        # Create the CUDA context before the agent, so the UCX worker is GPU-capable.
        torch.zeros(1, device=f"cuda:{self.gpu}")
        torch.cuda.synchronize()
        config = nixl_agent_config(backends=[cfg["backend"]])
        self.agent = nixl_agent(self.name, config)
        self.checks = []
        self.step = 0
        self.skip = ""
        if self.owner:
            self.vmm = _Vmm(self.gpu, cfg["handle"], cfg["chunk_units"])
            self.chunk = self.vmm.chunk
            self.regs = []
        else:
            self.info = {}
            self.chunk = 0
            self.buf = None
            self.remote = None
            self.prepared = {}
            # Requests that never completed. They stay referenced, never released.
            self.stuck = []
            self.written = []
            self.refill = []

    def check(self, label, ok):
        self.checks.append((label, bool(ok)))

    # ----- sync -----

    def _publish(self, name, data):
        path = os.path.join(self.exch, name)
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)

    def _await(self, name):
        path = os.path.join(self.exch, name)
        deadline = time.monotonic() + SYNC_TIMEOUT_S
        while not os.path.exists(path):
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self.name}: no {name} in {SYNC_TIMEOUT_S:.0f}s")
            time.sleep(0.01)
        with open(path, "rb") as f:
            return f.read()

    def sync(self, tag, load=False):
        """Barrier. The owner verifies the peers' writes and refills, then releases them.

        `load` names the peers that load the owner's metadata after the barrier;
        True means every peer.
        """
        self.step += 1
        key = f"{self.step:03d}-{tag}"
        loaders = self.names[1:] if load is True else tuple(load or ())
        if self.owner:
            for rank in range(1, len(self.names)):
                self._verify(json.loads(self._await(f"{key}.{rank}")))
            if loaders:
                self._publish(f"{key}.md", self.agent.get_agent_metadata())
            info = {
                "chunk": self.chunk,
                "va": {name: a.va for name, a in self.vmm.arenas.items()},
                "skip": self.skip,
            }
            self._publish(f"{key}.go", json.dumps(info).encode())
            return
        sent = {"writes": self.written, "refill": self.refill}
        self.written, self.refill = [], []
        self._publish(f"{key}.{self.rank}", json.dumps(sent).encode())
        self.info = json.loads(self._await(f"{key}.go"))
        self.chunk = self.info["chunk"]
        self.skip = self.info["skip"]
        if self.name in loaders:
            self._load(f"{key}.md", tag)

    def _verify(self, sent):
        for label, arena, offset, points in sent["writes"]:
            va = self.vmm.arenas[arena].va + offset
            landed = all(self.vmm.matches(va + at, WRITE) for at in points)
            self.check(label, landed)
        for arena, index, pattern in sent["refill"]:
            self.vmm.fill(arena, index, pattern)

    def finish(self):
        """Last barrier. The owner returns every rank's checks."""
        self.step += 1
        key = f"{self.step:03d}-done"
        if self.peer:
            sent = {"writes": self.written, "refill": [], "checks": self.checks}
            self._publish(f"{key}.{self.rank}", json.dumps(sent).encode())
            self._await(f"{key}.go")
            return []
        peer_checks = []
        for rank in range(1, len(self.names)):
            sent = json.loads(self._await(f"{key}.{rank}"))
            self._verify(sent)
            name = self.names[rank]
            peer_checks += [(f"{name}:{label}", ok) for label, ok in sent["checks"]]
        self._publish(f"{key}.go", b"{}")
        return [(f"owner:{label}", ok) for label, ok in self.checks] + peer_checks

    # ----- owner -----

    def setup(self, arena, chunks, mapped, load=True):
        """Owner reserves and registers an arena; then the peers load."""
        if self.owner:
            self.reserve(arena, chunks, mapped)
            self.register("initial", arena)
        self.sync("initial", load=load)

    def reserve(self, arena, chunks, mapped=(), after=None):
        self.vmm.reserve(arena, chunks, after)
        for index in mapped:
            self.vmm.map(arena, index, OLD)

    def register(self, label, arena, first=0, count=None, expect_ok=True):
        a = self.vmm.arenas[arena]
        start = a.va + first * self.chunk
        length = a.size - first * self.chunk if count is None else count * self.chunk
        descs = self.agent.get_reg_descs([(start, length, self.gpu, "")], "VRAM")
        try:
            reg = self.agent.register_memory(descs)
            error = "" if reg is not None else "returned None"
        except Exception as exc:
            reg, error = None, f"{type(exc).__name__}: {exc}"
        ok = reg is not None
        detail = "ok" if ok else error
        if not expect_ok:
            detail = "registered" if ok else "rejected"
        self.check(f"{label}:register_{arena}({detail})", ok == expect_ok)
        self.regs.append(reg)
        return len(self.regs) - 1

    def deregister(self, label, reg, expect_ok=True):
        try:
            self.agent.deregister_memory(self.regs[reg])
            ok, detail = True, "deregistered"
        except Exception as exc:
            ok, detail = False, type(exc).__name__
        self.check(f"{label}:deregister({detail})", ok == expect_ok)

    def retired(self, label, arena, pattern):
        kept = self.vmm.retired_matches(arena, pattern)
        self.check(f"{label}:retired_{arena}_keeps_{_hex(pattern)}", kept)

    def change(self, label, action, arena, index, pattern=None, load=False):
        """Owner maps, remaps, or unmaps one chunk and registers the arena again."""
        if self.owner:
            if action == "unmap":
                self.vmm.unmap(arena, index)
            else:
                getattr(self.vmm, action)(arena, index, pattern)
            self.register(label, arena)
        self.sync(label, load=load)

    def checked(self, label, arena, layout):
        """Peers check the layout of an arena; then the owner verifies their writes."""
        if self.peer:
            self.layout(label, arena, layout)
        self.sync(f"{label}_checked")

    # ----- peer -----

    def _load(self, name, tag):
        try:
            remote = self.agent.add_remote_agent(self._await(name))
        except Exception as exc:
            self.check(f"{tag}:load({type(exc).__name__}: {exc})", False)
            return
        remote = remote.decode() if isinstance(remote, bytes) else remote
        if self.remote is not None:
            self.check(f"{tag}:reload_keeps_name({remote})", remote == self.remote)
        self.remote = remote
        if self.buf is None:
            size = LOCAL_CHUNKS * self.chunk
            self.buf = torch.zeros(size, dtype=torch.uint8, device=f"cuda:{self.gpu}")
            self.agent.register_memory(self.agent.get_reg_descs(self.buf))

    def leave(self):
        """This peer stops; its agent goes away. It still attends the syncs."""
        self.agent = None
        self.remote = None

    def _fill_local(self, size, pattern):
        if self.buf is not None:
            self.buf[:size].fill_(pattern)
            torch.cuda.synchronize()

    def _local_is(self, offset, pattern):
        if self.buf is None:
            return False
        return bool((self.buf[offset : offset + 8] == pattern).all())

    def _local_pattern(self):
        for pattern in PATTERNS:
            if self._local_is(0, pattern) and self._local_is(PAYLOAD - 8, pattern):
                return pattern
        return "mixed"

    def _prepare(self, op, arena, offset, length):
        if self.remote is None:
            return None, "NO_REMOTE"
        va = self.info["va"][arena] + offset
        remote = self.agent.get_xfer_descs([(va, length, OWNER_GPU)], "VRAM")
        local = self.agent.get_xfer_descs(self.buf[:length])
        try:
            handle = self.agent.initialize_xfer(op, local, remote, self.remote, b"vmm")
        except Exception as exc:
            return None, f"PREP_ERR:{type(exc).__name__}"
        return handle, ""

    def _post(self, handle):
        """DONE, ERR, ERR:<exception>, or TIMEOUT."""
        deadline = time.monotonic() + XFER_TIMEOUT_S
        try:
            state = self.agent.transfer(handle)
            while state not in ("DONE", "ERR") and time.monotonic() < deadline:
                state = self.agent.check_xfer_state(handle)
        except Exception as exc:
            state = f"ERR:{type(exc).__name__}"
        if state != "DONE" and not _failed_cleanly(state):
            self.stuck.append(handle)
            return "TIMEOUT"
        try:
            self.agent.release_xfer_handle(handle)
        except Exception:
            self.stuck.append(handle)
        return state

    def _xfer(self, op, arena, offset, length):
        handle, error = self._prepare(op, arena, offset, length)
        return error if handle is None else self._post(handle)

    def read(self, label, arena, index, expect):
        """READ one chunk. `expect` is its pattern, or None for a hole that must fail."""
        self._fill_local(PAYLOAD, 0)
        state = self._xfer("READ", arena, index * self.chunk, PAYLOAD)
        where = f"{label}:{arena}{index}:{state}"
        if expect is None:
            self.check(f"{where}:hole", _failed_cleanly(state))
            return
        got = self._local_pattern() if state == "DONE" else None
        self.check(f"{where}:{_hex(got)}", state == "DONE" and got == expect)

    def write(self, label, arena, index, restore):
        """WRITE one chunk. The owner checks it landed, then refills `restore`."""
        self._fill_local(PAYLOAD, WRITE)
        offset = index * self.chunk
        state = self._xfer("WRITE", arena, offset, PAYLOAD)
        where = f"{label}:write_{arena}{index}"
        self.check(f"{where}:{state}", state == "DONE")
        if state == "DONE":
            self.written.append((f"{where}:landed", arena, offset, [0, PAYLOAD - 8]))
        if restore is not None:
            self.refill.append((arena, index, restore))

    def prepare(self, tag, arena, index):
        """Prepare a READ of one chunk now and post it later with post_prepared."""
        self.prepared[tag] = self._prepare("READ", arena, index * self.chunk, PAYLOAD)

    def post_prepared(self, label, tag, expect):
        handle, error = self.prepared.pop(tag)
        self._fill_local(PAYLOAD, 0)
        state = error if handle is None else self._post(handle)
        got = self._local_pattern() if state == "DONE" else None
        where = f"{label}:prepared_{tag}:{state}:{_hex(got)}"
        self.check(where, state == "DONE" and got == expect)

    def span(self, label, arena, first, patterns):
        """READ and WRITE one request across consecutive mapped chunks.

        It starts halfway into the first chunk and ends halfway into the last one.
        """
        c = self.chunk
        start, length = first * c + c // 2, (len(patterns) - 1) * c
        points = _span_points(c, length, patterns)
        where = f"{label}:span_{arena}{first}-{first + len(patterns) - 1}"
        self._fill_local(length, 0)
        state = self._xfer("READ", arena, start, length)
        good = all(self._local_is(at, p) for at, p in points.items())
        self.check(f"{where}:read:{state}", state == "DONE" and good)
        self._fill_local(length, WRITE)
        state = self._xfer("WRITE", arena, start, length)
        self.check(f"{where}:write:{state}", state == "DONE")
        if state == "DONE":
            self.written.append((f"{where}:landed", arena, start, sorted(points)))
        for offset, pattern in enumerate(patterns):
            self.refill.append((arena, first + offset, pattern))

    def _must_fail(self, where, arena, start, length):
        """READ and WRITE a request that must return a NIXL error."""
        self._fill_local(length, 0)
        state = self._xfer("READ", arena, start, length)
        self.check(f"{where}:read:{state}", _failed_cleanly(state))
        self._fill_local(length, WRITE)
        state = self._xfer("WRITE", arena, start, length)
        self.check(f"{where}:write:{state}", _failed_cleanly(state))

    def into_hole(self, label, arena, mapped, hole, pattern):
        """A chunk-long request, half in a hole and half in a neighbouring mapped chunk."""
        c = self.chunk
        start = min(mapped, hole) * c + c // 2
        self._must_fail(f"{label}:{arena}{mapped}_into_hole{hole}", arena, start, c)
        self.refill.append((arena, mapped, pattern))

    def cross(self, label, a_last, b_first):
        """A chunk-long request from the middle of a's last chunk into b."""
        c = self.chunk
        self._must_fail(f"{label}:a_into_b", "a", c + c // 2, c)
        self.refill += [("a", 1, a_last), ("b", 0, b_first)]

    def layout(self, label, arena, layout):
        """Check every position: index -> pattern, or None for a hole.

        Every chunk is read on its own and every hole must return a NIXL error.
        The longest run of mapped chunks is read and written as one request, and a
        request half in each hole and half in a neighbouring mapped chunk must fail.
        """
        for index, expect in sorted(layout.items()):
            self.read(label, arena, index, expect)
        run = _longest_run(layout)
        if len(run) >= 2:
            self.span(label, arena, run[0], [layout[i] for i in run])
        for hole in sorted(i for i, expect in layout.items() if expect is None):
            for mapped in (hole - 1, hole + 1):
                if layout.get(mapped) is not None:
                    self.into_hole(label, arena, mapped, hole, layout[mapped])
                    break


# ===== Registration rules =====
# These do not change memory after the peer has loaded, or test registerMem itself.


def _never_remap(w):
    """Register, load, READ, and WRITE with no remap behave as today."""
    w.setup("a", 2, (0, 1))
    w.checked("initial", "a", {0: OLD, 1: OLD})
    if w.peer:
        w.write("initial", "a", 1, OLD)


def _register_with_holes(w):
    """A registration over a hole succeeds. The hole returns a NIXL error."""
    w.setup("a", 3, (0, 2))
    w.checked("initial", "a", {0: OLD, 1: None, 2: OLD})


def _exact_register_is_remap(w):
    """A second registerMem of the same range after a remap updates the same entry.

    The peer loads again. One deregister succeeds and a second one fails.
    """
    if w.owner:
        w.reserve("a", 2, (0, 1))
        w.register("first", "a")
    w.sync("first", load=True)
    if w.owner:
        w.vmm.remap("a", 0, NEW)
        w.register("second", "a")
    w.sync("second", load=True)
    if w.owner:
        w.deregister("once", 1)
        w.deregister("twice", 0, expect_ok=False)


def _subrange_register_straddle_rejected(w):
    """Registering chunks 1-2 against a registration of chunks 0-1 fails."""
    w.reserve("a", 3, (0, 1, 2))
    w.register("first", "a", 0, 2)
    w.register("straddle", "a", 1, 2, expect_ok=False)


# ===== Reload only =====
# The second loadRemote is the subject of the test.


def _identical_reload_noop(w):
    """Loading the same metadata again keeps the remote name and prepared requests."""
    w.setup("a", 1, (0,))
    if w.peer:
        w.prepare("before", "a", 0)
    w.sync("prepared")
    w.sync("same_md", load=True)
    if w.peer:
        w.post_prepared("same_md", "before", OLD)
        w.read("same_md", "a", 0, OLD)


# ===== Changes after the peer has loaded =====
# Each takes `reload`. With False the peer keeps the metadata of its first load and
# must follow every change. With True every registerMem is followed by loadRemote,
# which must be a no-op. A peer that joins later always loads once.

# ----- without another loadRemote -----


def _register_empty_then_map(w, reload):
    """Register a range with nothing mapped, load it, then map chunk 0.

    Chunk 0 transfers and chunk 1 stays a hole.
    """
    w.setup("a", 2, ())
    w.change("map0", "map", "a", 0, OLD, load=reload)
    w.checked("map0", "a", {0: OLD, 1: None})


def _remap_in_place(w, reload):
    """After a remap the peer reaches the new allocation; the retired one is untouched."""
    w.setup("a", 2, (0, 1))
    w.checked("initial", "a", {0: OLD, 1: OLD})
    w.change("remap0", "remap", "a", 0, NEW, load=reload)
    w.checked("remap0", "a", {0: NEW, 1: OLD})
    if w.peer:
        w.write("remap0", "a", 0, NEW)
    w.sync("written")
    if w.owner:
        w.retired("remap0", "a", OLD)


def _prepared_request_rebuilt(w, reload):
    """A READ prepared before a remap reaches the new allocation when posted after it."""
    w.setup("a", 1, (0,))
    if w.peer:
        w.prepare("before", "a", 0)
    w.sync("prepared")
    w.change("remap0", "remap", "a", 0, NEW, load=reload)
    if w.peer:
        w.read("remap0", "a", 0, NEW)
        w.post_prepared("remap0", "before", NEW)


def _different_tuple_shared_physical(w, reload):
    """One allocation is mapped at a1 and at b0, in two registrations.

    A write through a1 shows through b0. The owner then refills the shared allocation
    with 0x3C, which no request writes. After a1 is remapped, b0 stays on the shared
    allocation, and a write through b0 does not reach a1.
    """
    if w.owner:
        w.reserve("a", 2, (0, 1))
        w.reserve("b", 2, (1,))
        w.vmm.alias("b", 0, "a", 1)
        w.register("initial", "a")
        w.register("initial", "b")
    w.sync("initial", load=True)
    if w.peer:
        w.write("initial", "a", 1, REMAPPED)
        w.read("initial", "b", 0, WRITE)
    w.sync("shared")
    w.change("remap_a1", "remap", "a", 1, NEW, load=reload)
    # Requests through b reach the shared allocation, so it is checked before them.
    w.checked("remap_a1", "a", {0: OLD, 1: NEW})
    if w.owner:
        w.retired("remap_a1", "a", REMAPPED)
    w.checked("remap_a1", "b", {0: REMAPPED, 1: OLD})
    if w.peer:
        w.write("remap_a1", "b", 0, REMAPPED)
        w.read("remap_a1", "a", 1, NEW)


def _grow_into_hole(w, reload):
    """Map a hole; the new chunk and a request across both chunks transfer."""
    w.setup("a", 2, (0,))
    w.checked("initial", "a", {0: OLD, 1: None})
    w.change("map1", "map", "a", 1, NEW, load=reload)
    w.checked("map1", "a", {0: OLD, 1: NEW})


def _shrink(w, reload):
    """Unmap a chunk; the kept chunk transfers and the unmapped one fails."""
    w.setup("a", 2, (0, 1))
    w.checked("initial", "a", {0: OLD, 1: OLD})
    w.change("unmap1", "unmap", "a", 1, load=reload)
    w.checked("unmap1", "a", {0: OLD, 1: None})


def _hole_grow_unmap_remap(w, reload):
    """Fill the hole, remap chunk 0 twice, unmap chunk 1, then map it again.

    After each step the layout matches and each retired allocation is untouched.
    """
    w.setup("a", 2, (0,))
    w.checked("initial", "a", {0: OLD, 1: None})
    w.change("map1", "map", "a", 1, NEW, load=reload)
    w.checked("map1", "a", {0: OLD, 1: NEW})
    w.change("remap0", "remap", "a", 0, NEW, load=reload)
    w.checked("remap0", "a", {0: NEW, 1: NEW})
    if w.owner:
        w.retired("remap0", "a", OLD)
    w.change("remap0_again", "remap", "a", 0, REMAPPED, load=reload)
    w.checked("remap0_again", "a", {0: REMAPPED, 1: NEW})
    if w.owner:
        w.retired("remap0_again", "a", NEW)
    w.change("unmap1", "unmap", "a", 1, load=reload)
    w.checked("unmap1", "a", {0: REMAPPED, 1: None})
    w.change("map1_again", "map", "a", 1, OLD, load=reload)
    w.checked("map1_again", "a", {0: REMAPPED, 1: OLD})


def _long_lifetime_sequence(w, reload):
    """Twelve changes on one five-chunk buffer.

    After each change the layout matches and each retired allocation is untouched.
    """
    steps = [
        ("map1", "map", 1, NEW),
        ("remap0", "remap", 0, NEW),
        ("map2", "map", 2, NEW),
        ("remap3", "remap", 3, REMAPPED),
        ("map4", "map", 4, NEW),
        ("remap1", "remap", 1, REMAPPED),
        ("remap2", "remap", 2, REMAPPED),
        ("unmap2", "unmap", 2, None),
        ("remap4", "remap", 4, REMAPPED),
        ("map2_again", "map", 2, OLD),
        ("remap0_again", "remap", 0, REMAPPED),
        ("unmap4", "unmap", 4, None),
    ]
    state = {0: OLD, 3: OLD}
    w.setup("a", 5, (0, 3))
    w.checked("initial", "a", {i: state.get(i) for i in range(5)})
    for label, action, index, pattern in steps:
        retired = state.get(index) if action == "remap" else None
        if action == "unmap":
            del state[index]
        else:
            state[index] = pattern
        w.change(label, action, "a", index, pattern, load=reload)
        w.checked(label, "a", {i: state.get(i) for i in range(5)})
        if w.owner and retired is not None:
            w.retired(label, "a", retired)


def _sparse_grow_and_change(w, reload):
    """Only the ends of six chunks are mapped; fill, remap, and unmap across them.

    Each step changes several chunks before one registerMem: map a middle chunk and
    remap the last, then unmap the first and map another middle one, then map the
    rest. After each step the layout matches; after the remap the retired chunk is
    untouched.
    """
    w.setup("a", 6, (0, 5))
    w.checked("ends", "a", {0: OLD, 1: None, 2: None, 3: None, 4: None, 5: OLD})
    if w.owner:
        w.vmm.map("a", 2, NEW)
        w.vmm.remap("a", 5, NEW)
        w.register("fill_mid_remap_end", "a")
    w.sync("fill_mid_remap_end", load=reload)
    layout = {0: OLD, 1: None, 2: NEW, 3: None, 4: None, 5: NEW}
    w.checked("fill_mid_remap_end", "a", layout)
    if w.owner:
        w.retired("fill_mid_remap_end", "a", OLD)
        w.vmm.unmap("a", 0)
        w.vmm.map("a", 3, NEW)
        w.register("drop_start_grow_mid", "a")
    w.sync("drop_start_grow_mid", load=reload)
    layout = {0: None, 1: None, 2: NEW, 3: NEW, 4: None, 5: NEW}
    w.checked("drop_start_grow_mid", "a", layout)
    if w.peer:
        w.write("drop_start_grow_mid", "a", 3, NEW)
    w.sync("written")
    if w.owner:
        for index in (0, 1, 4):
            w.vmm.map("a", index, NEW)
        w.register("dense", "a")
    w.sync("dense", load=reload)
    w.checked("dense", "a", {index: NEW for index in range(6)})


def _span_into_hole(w, reload):
    """A request from a mapped chunk into a hole fails; once mapped, it succeeds."""
    w.setup("a", 3, (0, 1))
    if w.peer:
        w.into_hole("initial", "a", 1, 2, OLD)
    w.sync("initial_checked")
    w.change("fill2", "map", "a", 2, NEW, load=reload)
    if w.peer:
        w.span("fill2", "a", 1, [OLD, NEW])


# ----- with another loadRemote -----


def _subrange_register_updates(w, reload):
    """registerMem of a mapped chunk inside the registration updates it."""
    w.setup("a", 3, (0,))
    w.checked("initial", "a", {0: OLD, 1: None})
    if w.owner:
        w.vmm.map("a", 1, NEW)
        w.register("contained", "a", 1, 1)
    w.sync("contained", load=reload)
    w.checked("contained", "a", {0: OLD, 1: NEW, 2: None})
    if w.peer:
        w.write("contained", "a", 1, NEW)


def _grow_prefix(w, reload):
    """Only chunk 0 of four is mapped. Grow to two chunks, then to four.

    The last step maps two chunks before one registerMem. The new chunks transfer,
    and so does a request across all four.
    """
    w.setup("a", 4, (0,))
    w.checked("prefix1", "a", {0: OLD, 1: None, 2: None, 3: None})
    w.change("grow2", "map", "a", 1, NEW, load=reload)
    w.checked("grow2", "a", {0: OLD, 1: NEW, 2: None, 3: None})
    if w.owner:
        for index in (2, 3):
            w.vmm.map("a", index, NEW)
        w.register("grow4", "a")
    w.sync("grow4", load=reload)
    w.checked("grow4", "a", {0: OLD, 1: NEW, 2: NEW, 3: NEW})
    if w.peer:
        w.write("grow4", "a", 3, NEW)


def _span_chunks(w, reload):
    """A request across three chunks, before and after the middle one is remapped."""
    w.setup("a", 4, (0, 1, 2, 3))
    if w.peer:
        w.span("initial", "a", 0, [OLD, OLD, OLD])
    w.sync("initial_checked")
    w.change("remap1", "remap", "a", 1, NEW, load=reload)
    if w.peer:
        w.span("remap1", "a", 0, [OLD, NEW, OLD])
    w.sync("remap1_checked")
    if w.owner:
        w.retired("remap1", "a", OLD)


def _change_chunk_shape(w, reload):
    """Two single-chunk allocations become one over both, then split back.

    After each step every chunk, and a request across all four, reaches the
    current allocations.
    """
    w.setup("a", 4, (0, 1, 2, 3))
    if w.owner:
        for index in (1, 2):
            w.vmm.unmap("a", index)
        w.vmm.map("a", 1, NEW, count=2)
        w.register("merge1_2", "a")
    w.sync("merge1_2", load=reload)
    w.checked("merge1_2", "a", {0: OLD, 1: NEW, 2: NEW, 3: OLD})
    if w.owner:
        # Chunk 1 is one allocation over chunks 1 and 2, so this unmaps both.
        w.vmm.unmap("a", 1)
        for index in (1, 2):
            w.vmm.map("a", index, OLD)
        w.register("split1_2", "a")
    w.sync("split1_2", load=reload)
    w.checked("split1_2", "a", {0: OLD, 1: OLD, 2: OLD, 3: OLD})


def _multi_peer_update(w, reload):
    """Two peers on their own chunks; a third joins; one leaves.

    Each peer follows every change to its chunk, the peer that joins late loads
    once, and the remaining peers keep working after one leaves.
    """
    both = ("peerA", "peerB")
    w.setup("a", 3, (0, 1), load=both)
    if w.name == "peerA":
        w.read("initial", "a", 0, OLD)
    if w.name == "peerB":
        w.read("initial", "a", 1, OLD)
    w.sync("initial_checked")
    w.change("remap1", "remap", "a", 1, NEW, load=reload and both)
    if w.name == "peerA":
        w.read("remap1", "a", 0, OLD)
    if w.name == "peerB":
        w.read("remap1", "a", 1, NEW)
    w.sync("remap1_checked")
    w.change("C_joins", "map", "a", 2, NEW, load=("peerC",) + (both if reload else ()))
    if w.name == "peerC":
        w.read("C_joins", "a", 2, NEW)
    w.sync("C_joins_checked")
    if w.name == "peerB":
        w.leave()
    w.sync("B_left")
    w.change("unmap1", "unmap", "a", 1, load=reload and ("peerA", "peerC"))
    if w.name == "peerA":
        w.layout("unmap1", "a", {0: OLD, 1: None, 2: NEW})
    if w.name == "peerC":
        w.read("unmap1", "a", 2, NEW)


def _different_tuple_adjacent(w, reload):
    """b is reserved where a ends. A request from a into b is rejected, and
    remapping a's last chunk leaves b unchanged."""
    if w.owner:
        w.reserve("a", 2, (0, 1))
        try:
            w.reserve("b", 2, (0, 1), after="a")
            w.register("initial", "a")
            w.register("initial", "b")
        except _NotAdjacent as exc:
            w.skip = f"adjacent reserve: {exc}"
    w.sync("initial", load=True)
    if w.skip:
        return
    if w.peer:
        w.cross("initial", OLD, OLD)
    w.sync("initial_checked")
    w.change("remap_a1", "remap", "a", 1, NEW, load=reload)
    if w.peer:
        w.layout("remap_a1", "a", {0: OLD, 1: NEW})
        w.layout("remap_a1", "b", {0: OLD, 1: OLD})
        w.cross("remap_a1", NEW, OLD)


# ===== Mixed =====


def _one_peer_reloads(w):
    """After a remap only peerA loads again. Both peers read the new allocation."""
    w.setup("a", 1, (0,))
    w.change("remap0", "remap", "a", 0, NEW, load=("peerA",))
    if w.peer:
        w.read("remap0", "a", 0, NEW)


def _scenario(fn, reload=None, peers=("peerA",)):
    return pytest.param(fn, reload, peers, id=fn.__name__.lstrip("_"))


SCENARIOS = [
    # Registration rules
    _scenario(_never_remap),
    _scenario(_register_with_holes),
    _scenario(_exact_register_is_remap),
    _scenario(_subrange_register_straddle_rejected, peers=()),
    # Reload only
    _scenario(_identical_reload_noop),
    # Without another loadRemote
    _scenario(_register_empty_then_map, False),
    _scenario(_remap_in_place, False),
    _scenario(_prepared_request_rebuilt, False),
    _scenario(_different_tuple_shared_physical, False),
    _scenario(_grow_into_hole, False),
    _scenario(_shrink, False),
    _scenario(_hole_grow_unmap_remap, False),
    _scenario(_long_lifetime_sequence, False),
    _scenario(_sparse_grow_and_change, False),
    _scenario(_span_into_hole, False),
    # With another loadRemote
    _scenario(_subrange_register_updates, True),
    _scenario(_grow_prefix, True),
    _scenario(_span_chunks, True),
    _scenario(_change_chunk_shape, True),
    _scenario(_multi_peer_update, True, ("peerA", "peerB", "peerC")),
    _scenario(_different_tuple_adjacent, True),
    # Mixed
    _scenario(_one_peer_reloads, peers=("peerA", "peerB")),
]


def _vmm_worker(rank, scenario, reload, names, exch, cfg):
    w = _Rank(rank, names, exch, cfg)
    if reload is None:
        scenario(w)
    else:
        scenario(w, reload)
    checks = w.finish()
    if w.owner:
        if w.skip:
            with open(os.path.join(exch, "skip"), "w") as f:
                f.write(w.skip)
        failed = [label for label, ok in checks if not ok]
        assert not failed, f"{len(failed)} of {len(checks)} checks failed:\n" + (
            "\n".join(failed)
        )
    # Exit hard, skipping NIXL/UCX/CUDA destructors: a request that never completed
    # must not be released, and teardown in a spawned worker can segfault.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("scenario,reload,peers", SCENARIOS)
def test_vmm_dynamic_buffer(scenario, reload, peers, vmm_config, capfd):
    # One process per role, as in test_prep_mem_view. capfd.disabled() restores the
    # real fds: children writing from C under pytest's capture segfault on teardown.
    import torch.multiprocessing as mp

    names = ("owner", *peers)
    with tempfile.TemporaryDirectory(prefix="vmm_exch_") as exch:
        with capfd.disabled():
            mp.spawn(
                _vmm_worker,
                args=(scenario, reload, names, exch, vmm_config),
                nprocs=len(names),
                join=True,
            )
        skip = os.path.join(exch, "skip")
        if os.path.exists(skip):
            with open(skip) as f:
                pytest.skip(f.read())
