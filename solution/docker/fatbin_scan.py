"""List the SM architectures of every cubin / PTX entry embedded (NVIDIA fatbin containers) in the given ELF files.
Fatbin header: magic 0xBA55ED50 u32, version u16, headerSize u16, fatSize u64; entries follow:
kind u16 (1 = PTX, 2 = ELF/cubin), u16, headerSize u32, paddedPayloadSize u64, u32, payloadSize u32, u32, arch u32 ..."""
import mmap, struct, sys, collections, os
MAGIC = b"\x50\xED\x55\xBA"
for path in sys.argv[1:]:
    counts = collections.Counter()
    nfat = 0
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        pos = mm.find(MAGIC)
        while pos >= 0:
            magic, ver, hsz, fsz = struct.unpack_from("<IHHQ", mm, pos)
            if ver != 1 or hsz != 16 or fsz == 0 or pos + hsz + fsz > len(mm):
                pos = mm.find(MAGIC, pos + 4); continue
            nfat += 1
            p, end = pos + hsz, pos + hsz + fsz
            while p + 32 <= end:
                kind, _u, ehsz, padded = struct.unpack_from("<HHIQ", mm, p)
                if kind not in (1, 2) or ehsz < 32 or padded == 0 or p + ehsz + padded > end + 8:
                    break
                arch = struct.unpack_from("<I", mm, p + 28)[0]
                counts[("PTX" if kind == 1 else "cubin", arch)] += 1
                p += ehsz + padded
            pos = mm.find(MAGIC, end)
    print(os.path.basename(path), "fatbins:", nfat, "size MB: %.0f" % (os.path.getsize(path) / 1e6))
    for (k, a), n in sorted(counts.items()):
        print("   %-5s sm_%-4d x %d" % (k, a, n))
