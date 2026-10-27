#!/usr/bin/env python3
"""Cold random-read bandwidth of the internal SSD at expert-sized reads.

Writes a scratch file with F_NOCACHE (so it never enters the unified buffer
cache), then preads random aligned blocks from it with F_NOCACHE at several
queue depths. F_NOCACHE does not stop every repeat read from hitting the cache,
so the disk0 rate from iostat is the number to use for the streaming ceiling:
    tok/s ceiling = SSD GB/s / streamed GB per token

usage: ssd_read_bench.py [--file PATH] [--size-gb 16] [--block-kb 512]
"""
import argparse
import fcntl
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48  # <sys/fcntl.h>


def write_scratch(path, size):
    chunk = os.urandom(64 << 20)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    written = 0
    while written < size:
        written += os.write(fd, chunk)
    os.fsync(fd)
    os.close(fd)


def disk0_mb():
    out = subprocess.check_output(["iostat", "-Id", "disk0"]).decode().splitlines()
    return float(out[2].split()[2])


def bench(path, size, block, qd, seconds):
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    nblocks = size // block

    def worker(seed):
        rng = random.Random(seed)
        n, deadline = 0, time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            n += len(os.pread(fd, block, rng.randrange(nblocks) * block))
        return n

    d0, t0 = disk0_mb(), time.perf_counter()
    with ThreadPoolExecutor(qd) as ex:
        total = sum(ex.map(worker, range(qd)))
    dt = time.perf_counter() - t0
    disk = (disk0_mb() - d0) * 1e6 / dt / 1e9
    os.close(fd)
    return total / dt / 1e9, disk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(os.getcwd(), ".ssd_read_bench.tmp"))
    ap.add_argument("--size-gb", type=float, default=16)
    ap.add_argument("--block-kb", type=int, nargs="+", default=[512, 1536])
    ap.add_argument("--qd", type=int, nargs="+", default=[1, 6, 24, 64])
    ap.add_argument("--seconds", type=float, default=5)
    args = ap.parse_args()

    size = int(args.size_gb * (1 << 30))
    print(f"writing {args.size_gb:.0f} GiB scratch file (F_NOCACHE): {args.file}")
    write_scratch(args.file, size)
    try:
        print(f"{'block':>8} " + " ".join(f"{'QD' + str(q):>13}" for q in args.qd)
              + "   (GB/s: app / disk0)")
        for kb in args.block_kb:
            rates = [bench(args.file, size, kb * 1024, q, args.seconds) for q in args.qd]
            print(f"{kb:>6}KB " + " ".join(f"{a:6.2f}/{d:<6.2f}" for a, d in rates))
    finally:
        os.remove(args.file)


if __name__ == "__main__":
    main()
