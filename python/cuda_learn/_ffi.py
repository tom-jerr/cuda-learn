"""TVM FFI 加载层：dlopen libcuda_learn.so，缓存全局函数。

libcuda_learn.so 中每个 op 通过 TVM_FFI_STATIC_INIT_BLOCK (constructor) 在
dlopen 时把 "cuda_learn.<op>" 注册进 libtvm_ffi.so 的全局注册表，这里
用 tvm.get_global_func 取回。
"""

import os

import tvm
import tvm_ffi  # noqa: F401  # 确保 libtvm_ffi.so 先以 RTLD_GLOBAL 加载

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_LIB_PATH = os.environ.get(
    "CUDA_LEARN_LIB", os.path.join(_REPO_ROOT, "build", "libcuda_learn.so"))

if not os.path.exists(_LIB_PATH):
    raise FileNotFoundError(
        f"libcuda_learn.so not found at {_LIB_PATH}.\n"
        f"Build it first: cmake -B {_REPO_ROOT}/build && "
        f"cmake --build {_REPO_ROOT}/build -j")

# dlopen 插件，触发 constructor 里的 GlobalDef().def(...) 注册。
tvm.runtime.load_module(_LIB_PATH)

_FUNC_CACHE = {}


def get_func(name):
    f = _FUNC_CACHE.get(name)
    if f is None:
        f = tvm.get_global_func(name)
        if f is None:
            raise RuntimeError(
                f"global func {name!r} not registered after loading "
                f"{_LIB_PATH}; check the CUDA_LEARN_REGISTER call")
        _FUNC_CACHE[name] = f
    return f


def call(name, *args):
    """在 torch 当前流上调用全局函数。

    args 为 torch.Tensor（经 __dlpack__ 零拷贝）或 Python 标量。
    tvm_ffi.use_torch_stream() 把 torch 当前流写入 FFI 线程局部环境，
    C++ 侧 TVMFFIEnvGetStream 取回后在该流上 launch kernel，保证与
    torch 算子及 Event 计时同流。
    """
    with tvm_ffi.use_torch_stream():
        return get_func(name)(*args)
