#!/usr/bin/env python3
"""Hand-edited PTX for the texture kernels: keep NVIDIA's code up to a cut line, then call a native tail.

usage: make_ptx.py KERNEL OUT.ptx
"""
import os
import re
import sys
from pathlib import Path

# PTX extracted from nvngx_dlss.dll by kernels/tools/extract_dlss_ptx.py (kernels/build.sh does this)
PTX_DIR = Path(os.environ.get("D4R_DLSS_PTX_DIR", Path(__file__).resolve().parent.parent / "extracted" / "ptx"))

KERNELS = {
    # enc0: cut after the bar.warp.sync that ends the smem staging of the GEMM A operand
    "rrlite_enc0_4x4_mvhi_hdr_folded": dict(
        file="dlss-0185-00.ptx",
        cut_after=2548,
        cut_check="bar.warp.sync -1;",
        sust=True,
        extern=""".extern .func d4r_enc0_tail
(
	.param .b32 d4r_a0, .param .b64 d4r_a1, .param .b64 d4r_a2, .param .b32 d4r_a3,
	.param .b32 d4r_a4, .param .b32 d4r_a5, .param .b32 d4r_a6
)
;
""",
        tail="""{
	.param .b32 q0;
	st.param.b32 [q0+0], %r259;
	.param .b64 q1;
	st.param.b64 [q1+0], %rd106;
	.param .b64 q2;
	st.param.b64 [q2+0], %rd78;
	.param .b32 q3;
	st.param.b32 [q3+0], %r1066;
	.param .b32 q4;
	st.param.b32 [q4+0], %r1067;
	.param .b32 q5;
	st.param.b32 [q5+0], %r1230;
	.param .b32 q6;
	st.param.b32 [q6+0], %r1233;
	call.uni d4r_enc0_tail, (q0, q1, q2, q3, q4, q5, q6);
}
ret;

}
""",
    ),
    # dec0: the GEMM (from the B fragment loads to the e4m3 staging stores) becomes a native call; the
    # A loads before it are left dead. %r560 (shared base) is the only register of the range used later.
    "rrlite_dec0_4x4_folded": dict(
        file="dlss-0195-00.ptx",
        delete=(139, 587),
        delete_check=("mov.u32 %r17, %laneid;", "st.shared.v2.u16 [%r570+1168], {%rs63, %rs64};"),
        sust=True,
        extern=""".extern .func d4r_dec0_head
(
	.param .b32 d4r_a0, .param .b64 d4r_a1, .param .b64 d4r_a2, .param .b32 d4r_a3,
	.param .b32 d4r_a4, .param .b32 d4r_a5, .param .b32 d4r_a6
)
;
""",
        insert="""mov.u32 %r560, _ZZ22rrlite_dec0_4x4_foldedN6RRLite18Decoder0ParametersEE4smem;
{
	.param .b32 q0;
	st.param.b32 [q0+0], %r560;
	.param .b64 q1;
	st.param.b64 [q1+0], %rd22;
	.param .b64 q2;
	st.param.b64 [q2+0], %rd24;
	.param .b32 q3;
	st.param.b32 [q3+0], %r475;
	.param .b32 q4;
	st.param.b32 [q4+0], %r476;
	.param .b32 q5;
	st.param.b32 [q5+0], %r3;
	.param .b32 q6;
	st.param.b32 [q6+0], %r4;
	call.uni d4r_dec0_head, (q0, q1, q2, q3, q4, q5, q6);
}""",
    ),
    # post and downsample: only the formatted surface stores become native (inline format conversion
    # instead of ZLUDA's per-channel surface_formatted_store_bits calls)
    "rrlite_post_3_2_mvhi_hdr_folded": dict(file="dlss-0176-00.ptx", sust=True, extern=""),
    "rrlite_downsample_kernel_static_hdr": dict(file="dlss-0196-00.ptx", sust=True, extern=""),
    # preset K (DLSS 4): the output kernel's 17 surface stores (13 formatted, 4 raw 32-bit)
    "hiluma_engine_output_depthinv_mvhi_hdr_max_v2_rel": dict(file="dlss-0078-00.ptx", sust=True, rz_round=True, extern=""),
    "hiluma_engine_input_depthinv_mvhi_hdr_v2_rel": dict(file="dlss-0062-00.ptx", sust=True, rz_round=True, extern=""),
}


SUST_EXTERN = """.extern .func d4r_sust_v2b16
(
	.param .b64 d4r_s0, .param .b32 d4r_s1, .param .b32 d4r_s2, .param .b16 d4r_s3, .param .b16 d4r_s4
)
;
"""
SUSTP_EXTERN = """.extern .func d4r_sust_p_v4b32
(
	.param .b64 d4r_t0, .param .b32 d4r_t1, .param .b32 d4r_t2, .param .b32 d4r_t3, .param .b32 d4r_t4,
	.param .b32 d4r_t5, .param .b32 d4r_t6
)
;
"""
SUSTP_RE = re.compile(r"^sust\.p\.2d\.v4\.b32\.zero \[(%rd\d+), \{(%r\d+),(%r\d+)\}\], \{(%f\d+),(%f\d+),(%f\d+),(%f\d+)\};$")
SUSTB32_EXTERN = """.extern .func d4r_sust_b32
(
	.param .b64 d4r_w0, .param .b32 d4r_w1, .param .b32 d4r_w2, .param .b32 d4r_w3
)
;
"""
SUSTB32_RE = re.compile(r"^sust\.b\.2d\.b32\.zero \[(%rd\d+), \{(%r\d+),(%r\d+)\}\], \{(%r\d+)\};$")
SUST_RE = re.compile(r"^sust\.b\.2d\.v2\.b16\.zero \[(%rd\d+), \{(%r\d+),(%r\d+)\}\], \{(%rs\d+),(%rs\d+)\};$")


SUST_KINDS = set(os.environ.get("D4R_SUST_KINDS", "b16,p,b32").split(","))


def replace_sust(lines):
    out, n = [], {"b16": 0, "p": 0, "b32": 0}
    for line in lines:
        mb = SUSTB32_RE.match(line.strip()) if "b32" in SUST_KINDS else None
        if mb:
            s, x, y, v = mb.groups()
            out.append(f"""{{
	.param .b64 u0;
	st.param.b64 [u0+0], {s};
	.param .b32 u1;
	st.param.b32 [u1+0], {x};
	.param .b32 u2;
	st.param.b32 [u2+0], {y};
	.param .b32 u3;
	st.param.b32 [u3+0], {v};
	call d4r_sust_b32, (u0, u1, u2, u3);
}}""")
            n["b32"] += 1
            continue
        mp = SUSTP_RE.match(line.strip()) if "p" in SUST_KINDS else None
        if mp:
            s, x, y, *v = mp.groups()
            st = "\n".join(f"\t.param .b32 v{i};\n\tst.param.b32 [v{i}+0], {r};" for i, r in enumerate(v))
            out.append(f"""{{
	.param .b64 u0;
	st.param.b64 [u0+0], {s};
	.param .b32 u1;
	st.param.b32 [u1+0], {x};
	.param .b32 u2;
	st.param.b32 [u2+0], {y};
{st}
	call d4r_sust_p_v4b32, (u0, u1, u2, v0, v1, v2, v3);
}}""")
            n["p"] += 1
            continue
        m = SUST_RE.match(line.strip())
        if not m:
            out.append(line)
            continue
        s, x, y, a, b = m.groups()
        out.append(f"""{{
	.param .b64 u0;
	st.param.b64 [u0+0], {s};
	.param .b32 u1;
	st.param.b32 [u1+0], {x};
	.param .b32 u2;
	st.param.b32 [u2+0], {y};
	.param .b16 u3;
	st.param.b16 [u3+0], {a};
	.param .b16 u4;
	st.param.b16 [u4+0], {b};
	call d4r_sust_v2b16, (u0, u1, u2, u3, u4);
}}""")
        n["b16"] += 1
    return out, n


RZ_RE = re.compile(r"^add\.rz\.ftz\.f32 (%f\d+), (%f\d+), (%f\d+);$")
CVT_RE = re.compile(r"^cvt\.rzi\.f32\.f32 (%f\d+), (%f\d+);$")


def replace_rz_round(lines):
    """roundf idiom add.rz(x, copysign(0.5, x)) + cvt.rzi -> exact round-half-away in default rounding:
    t = trunc(x); r = |x - t| >= 0.5 ? t + 2 * copysign(0.5, x) : t (x - t is exact). Lets ZLUDA compile the
    kernel without the strict-FP (constrained) mode that one explicit rounding mode forces."""
    out, n, i = [], 0, 0
    while i < len(lines):
        m = RZ_RE.match(lines[i].strip())
        if m:
            d, x, sgn = m.groups()
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            c = CVT_RE.match(lines[j].strip()) if j < len(lines) else None
            if c and c.group(2) == d:
                r = c.group(1)
                out += [f"cvt.rzi.f32.f32 %rq{4*n}, {x};", f"sub.f32 %rq{4*n+1}, {x}, %rq{4*n};",
                        f"abs.f32 %rq{4*n+1}, %rq{4*n+1};", f"setp.ge.f32 %pq{n}, %rq{4*n+1}, 0f3F000000;",
                        f"add.f32 %rq{4*n+2}, {sgn}, {sgn};", f"add.f32 %rq{4*n+3}, %rq{4*n}, %rq{4*n+2};",
                        f"selp.f32 {r}, %rq{4*n+3}, %rq{4*n}, %pq{n};"]
                n += 1
                i = j + 1
                continue
            raise SystemExit(f"add.rz at line {i} not followed by its cvt.rzi")
        out.append(lines[i])
        i += 1
    return out, n


def module_file(name, k):
    """The extracted PTX module that defines NAME: the 310.7 file number, else any module that does
    (other DLSS versions number their modules differently)."""
    entry = re.compile(rf"\.entry\s+{re.escape(name)}\s*\(")
    known = PTX_DIR / k["file"]
    if known.exists() and entry.search(known.read_text()):
        return known
    for path in sorted(PTX_DIR.glob("*.ptx")):
        if entry.search(path.read_text()):
            return path
    sys.exit(f"no PTX module in {PTX_DIR} defines {name}")


def main():
    name, out = sys.argv[1], Path(sys.argv[2])
    k = KERNELS[name]
    lines = module_file(name, k).read_text().replace("\r\n", "\n").split("\n")
    if "cut_after" in k:
        cut = k["cut_after"]
        if lines[cut - 1].strip() != k["cut_check"]:
            sys.exit(f"line {cut} is {lines[cut - 1]!r}, expected {k['cut_check']!r}")
        head, tail = lines[:cut], k["tail"].split("\n")
    elif "delete" in k:
        a, b = k["delete"]
        if (lines[a - 1].strip(), lines[b - 1].strip()) != k["delete_check"]:
            sys.exit(f"lines {a}/{b} are {lines[a - 1]!r}/{lines[b - 1]!r}, expected {k['delete_check']}")
        head, tail = lines[:a - 1] + k["insert"].split("\n") + lines[b:], []
    else:
        head, tail = lines, []
    entry = next(i for i, l in enumerate(head) if l.startswith(".visible .entry"))
    body, n = replace_sust(head[entry:]) if k.get("sust") else (head[entry:], {"b16": 0, "p": 0, "b32": 0})
    if k.get("rz_round"):
        body, nrz = replace_rz_round(body)
        # temporaries for the rewrite, declared after the kernel's own .reg lines
        at = next(i for i, l in enumerate(body) if l.startswith(".reg"))
        body = body[:at] + [f".reg .f32 %rq<{4 * nrz + 1}>;", f".reg .pred %pq<{nrz + 1}>;"] + body[at:]
        n["rz"] = nrz
    extern = k["extern"] + (SUST_EXTERN if n["b16"] else "") + (SUSTP_EXTERN if n["p"] else "") + (SUSTB32_EXTERN if n["b32"] else "")
    text = "\n".join(head[:entry]) + "\n" + extern + "\n".join(body + tail)
    out.write_text(text)
    print(f"{out}: {len(text.splitlines())} lines, surface stores replaced: {n}")


if __name__ == "__main__":
    main()
