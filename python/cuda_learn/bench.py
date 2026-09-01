"""按测试函数名驱动的 benchmark 机制。

机制：tests 包中的每个 test_<op>() 用 @bench 装饰注册元数据
（输入工厂 / torch 参考实现 / flops / 容差 / warmup / iters）。
测试函数体只做 op 调用，正确性断言由本运行器在计时窗之外完成。

用法（先 source scripts/env.sh，并在 venv 中）：
  python -m cuda_learn.bench --list                # 列出全部测试
  python -m cuda_learn.bench test_gemm             # 正确性 + 计时
  python -m cuda_learn.bench gemm vector_add       # 支持唯一前缀/子串，多个名字
  python -m cuda_learn.bench --warmup 20 --iters 200 test_gemm
  python -m cuda_learn.bench --check-only test_gemm
"""

import argparse
import importlib
import pkgutil

import torch

BENCH_CASES = {}


class Case:
    def __init__(self, fn, make_inputs, ref, flops, rtol, atol, warmup, iters,
                 baselines, requires):
        self.fn = fn
        self.name = fn.__name__
        self.make_inputs = make_inputs
        self.ref = ref
        self.flops = flops
        self.rtol = rtol
        self.atol = atol
        self.warmup = warmup
        self.iters = iters
        self.baselines = baselines
        self.requires = requires


def bench(make_inputs, ref=None, flops=None, rtol=1e-4, atol=1e-4,
          warmup=10, iters=100, baselines=None, requires=None):
    """装饰 test_<op>() 并注册 benchmark 元数据。

    make_inputs: () -> tuple[torch.Tensor]，每次调用生成输入；
    ref: (*inputs) -> 参考输出（torch 实现），形状/结构与 op 输出一致；
    flops: (*inputs) -> float，每次调用的浮点运算数（报 TFLOPS 用）；
    baselines: {显示名: callable}，用相同输入和计时参数运行的库基线。
    requires: 可选的 ``() -> bool``，硬件不满足时整项显示为 SKIP。
    """
    def deco(fn):
        BENCH_CASES[fn.__name__] = Case(
            fn, make_inputs, ref, flops, rtol, atol, warmup, iters,
            dict(baselines or {}), requires)
        return fn
    return deco


def discover():
    """import cuda_learn.tests 下所有模块，收集 @bench 注册的用例。"""
    import cuda_learn.tests as tests
    for modinfo in pkgutil.walk_packages(tests.__path__, tests.__name__ + "."):
        importlib.import_module(modinfo.name)
    return BENCH_CASES


def resolve(names):
    """把命令行名字解析为测试函数名：支持全名、唯一前缀/子串。"""
    cases = discover()
    if not names:
        return sorted(cases)
    out = []
    for name in names:
        if name in cases:
            out.append(name)
            continue
        matches = [k for k in cases if k.startswith(name) or name in k]
        if not matches:
            raise KeyError(f"no test matching {name!r}; available: {sorted(cases)}")
        if len(matches) > 1:
            raise KeyError(f"{name!r} is ambiguous: {sorted(matches)}")
        out.append(matches[0])
    return out


def _as_tuple(v):
    return v if isinstance(v, (tuple, list)) else (v,)


def check_case(case, fn=None):
    """正确性：fn(*inputs) 与 torch 参考实现对比（计时窗之外）。"""
    torch.manual_seed(0)
    inputs = case.make_inputs()
    out = (case.fn if fn is None else fn)(*inputs)
    if case.ref is None:
        return "SKIP (no ref)"
    # 同一 seed → ref 拿到与 fn 完全相同的输入数据；
    # 单独生成一份避免原地修改类 op 污染参考计算。
    torch.manual_seed(0)
    ref_inputs = case.make_inputs()
    expected = case.ref(*ref_inputs)
    for i, (o, e) in enumerate(zip(_as_tuple(out), _as_tuple(expected))):
        try:
            torch.testing.assert_close(
                o.detach(), e.detach(), rtol=case.rtol, atol=case.atol)
        except AssertionError as exc:
            raise AssertionError(
                f"{case.name}: output {i} mismatch\n{exc}") from None
    return "PASS"


def bench_case(case, warmup=None, iters=None, fn=None):
    """性能：CUDA Event 计时（与 kernel 同流，一次 sync）。"""
    warmup = case.warmup if warmup is None else warmup
    iters = case.iters if iters is None else iters
    torch.manual_seed(0)
    inputs = case.make_inputs()
    fn = case.fn if fn is None else fn
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(*inputs)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    flops = case.flops(*inputs) if case.flops is not None else None
    return {
        "min": times[0],
        "median": times[len(times) // 2],
        "mean": sum(times) / len(times),
        "flops": flops,
    }


def _format_stats(label, stats, speedup=None):
    line = (f"    {label:16s} min {stats['min']:9.3f} ms | "
            f"median {stats['median']:9.3f} ms | mean {stats['mean']:9.3f} ms")
    if stats["flops"] is not None:
        flops_per_s = stats["flops"] / (stats["median"] * 1e-3)
        if flops_per_s >= 1e12:
            line += f" | {flops_per_s / 1e12:8.2f} TFLOPS"
        else:
            line += f" | {flops_per_s / 1e9:8.2f} GFLOPS"
    if speedup is not None:
        line += f" | cuda_learn speedup {speedup:6.2f}x"
    return line


def run_case(name, args):
    case = BENCH_CASES[name]
    if case.requires is not None and not case.requires():
        print(f"{name:32s} correctness: SKIP (hardware requirement)")
        return
    try:
        verdict = check_case(case)
    except (AssertionError, RuntimeError) as e:
        print(f"{name:32s} correctness: FAIL")
        print(f"    {e}")
        return
    print(f"{name:32s} correctness: {verdict}")
    valid_baselines = []
    for label, fn in case.baselines.items():
        try:
            baseline_verdict = check_case(case, fn)
        except (AssertionError, RuntimeError) as e:
            print(f"    {label:16s} correctness: SKIP ({e})")
            continue
        print(f"    {label:16s} correctness: {baseline_verdict}")
        valid_baselines.append((label, fn))
    if args.check_only:
        return
    stats = bench_case(case, args.warmup, args.iters)
    print(f"{name:32s} perf:")
    print(_format_stats("cuda_learn", stats))
    for label, fn in valid_baselines:
        try:
            baseline_stats = bench_case(
                case, args.warmup, args.iters, fn=fn)
        except RuntimeError as e:
            print(f"    {label:16s} perf: SKIP ({e})")
            continue
        speedup = baseline_stats["median"] / stats["median"]
        print(_format_stats(label, baseline_stats, speedup))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*",
                    help="test function names (full name or unique prefix/substring)")
    ap.add_argument("--list", action="store_true",
                    help="list all test names and exit")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--check-only", action="store_true",
                    help="只做正确性检查，不计时")
    args = ap.parse_args(argv)

    try:
        names = resolve(args.names)
    except KeyError as e:
        print(f"error: {e}")
        return 1
    if args.list:
        for n in names:
            print(n)
        return 0
    for name in names:
        run_case(name, args)
    return 0


if __name__ == "__main__":
    # python -m cuda_learn.bench 下本文件会以 __main__ 执行，而 tests 包里的
    # `from cuda_learn.bench import bench` 又会按规范名二次导入本文件，产生两个
    # 模块实例、两份 BENCH_CASES 注册表。跳转到规范名模块执行，避免注册表分裂。
    import cuda_learn.bench as _bench  # noqa: E402
    raise SystemExit(_bench.main())
