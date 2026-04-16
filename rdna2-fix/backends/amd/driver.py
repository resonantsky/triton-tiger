import functools
import os
import subprocess
import re
from pathlib import Path
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver
from triton.runtime.build import compile_module_from_src
from triton.tools.tensor_descriptor import TensorDescriptor

# Module-level globals populated by HIPUtils.__init__ once the C extension loads
PyTDMDescriptor = None
PyKernelArg = None
ARG_CONSTEXPR = None
ARG_KERNEL = None
ARG_TUPLE = None

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]


def _find_already_mmapped_dylib_on_linux(lib_name):
    import platform
    if platform.system() != 'Linux':
        return None

    # Use dl_iterate_phdr to walk through the list of shared libraries at runtime.
    # See https://www.man7.org/linux/man-pages/man3/dl_iterate_phdr.3.html for details.

    import ctypes
    from ctypes import c_char, c_int, c_size_t, c_void_p, c_char_p, POINTER

    class DlPhdrInfo(ctypes.Structure):
        _fields_ = [
            ('dlpi_addr', c_void_p),
            ('dlpi_name', c_char_p),
            # We don't care about the remaining fields.
        ]

    # callback_t must use POINTER(c_char) to avoid copying.
    callback_t = ctypes.CFUNCTYPE(c_int, POINTER(DlPhdrInfo), POINTER(c_size_t), POINTER(c_char))

    # Load libc and get the dl_iterate_phdr symbol.
    try:
        dl_iterate_phdr = ctypes.CDLL('libc.so.6').dl_iterate_phdr
    except Exception:
        return None
    # argtypes must use c_char_p to accept create_string_buffer.
    dl_iterate_phdr.argtypes = [callback_t, c_char_p]
    dl_iterate_phdr.restype = c_int

    max_path_length = 4096
    path = ctypes.create_string_buffer(max_path_length + 1)

    # Define callback to get the loaded dylib path.
    def callback(info, size, data):
        dlpi_name = info.contents.dlpi_name
        p = Path(os.fsdecode(dlpi_name))
        if lib_name in p.name:
            # Found the dylib; get its path.
            ctypes.memmove(data, dlpi_name, min(max_path_length, len(dlpi_name)))
            return 1
        return 0

    if dl_iterate_phdr(callback_t(callback), path):
        return os.fsdecode(ctypes.string_at(path))
    return None


@functools.lru_cache()
def _get_path_to_hip_runtime_dylib():
    if os.name == 'nt':
        lib_name = "amdhip64_7.dll"
    else:
        lib_name = "libamdhip64.so"

    # Explicit override via env var (must point to the actual DLL/SO)
    if env_libhip_path := knobs.amd.libhip_path:
        if os.path.exists(env_libhip_path):
            return env_libhip_path.replace('\\', '\\\\')
        raise RuntimeError(f"TRITON_LIBHIP_PATH '{env_libhip_path}' does not point to a valid {lib_name}")

    paths = []

    import site
    import sys
    # Prefer the venv-bundled _rocm_sdk_core DLL (bin/ on Windows)
    for sp in site.getsitepackages():
        sdk_dll = os.path.join(sp, '_rocm_sdk_core', 'bin', lib_name)
        if os.path.exists(sdk_dll):
            return sdk_dll.replace('\\', '\\\\')
        paths.append(sdk_dll)

    # PyTorch ships its own copy
    for sp in site.getsitepackages():
        torch_dll = os.path.join(sp, 'torch', 'lib', lib_name)
        if os.path.exists(torch_dll):
            return torch_dll.replace('\\', '\\\\')
        paths.append(torch_dll)

    # HIP_PATH / ROCM_PATH env var
    rocm_path = os.environ.get('HIP_PATH') or os.environ.get('ROCM_PATH')
    if rocm_path:
        lib_subdir = 'bin' if os.name == 'nt' else 'lib'
        hip_dll = os.path.join(rocm_path, lib_subdir, lib_name)
        if os.path.exists(hip_dll):
            return hip_dll.replace('\\', '\\\\')
        paths.append(hip_dll)

    # System-installed ROCm (common Windows install paths)
    if os.name == 'nt':
        for candidate_root in [
            r"C:\Program Files\AMD\ROCm\7.1",
            r"C:\Program Files\AMD\ROCm\7.0",
            r"C:\Program Files\AMD\ROCm\6.2",
        ]:
            sys_dll = os.path.join(candidate_root, 'bin', lib_name)
            if os.path.exists(sys_dll):
                return sys_dll.replace('\\', '\\\\')

    if os.name != 'nt':
        # If the shared object is already mmapped to address space, use it.
        mmapped_path = _find_already_mmapped_dylib_on_linux(lib_name)
        if mmapped_path and os.path.exists(mmapped_path):
            return mmapped_path

        try:
            libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
            locs = [line.split()[-1] for line in libs.splitlines() if line.strip().endswith(lib_name)]
            for loc in locs:
                if os.path.exists(loc):
                    return loc
                paths.append(loc)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        common_install_path = os.path.join('/opt/rocm/lib/', lib_name)
        if os.path.exists(common_install_path):
            return common_install_path
        paths.append(common_install_path)

    raise RuntimeError(f"cannot locate {lib_name} after attempted paths {paths}")


class HIPUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(HIPUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        libhip_path = _get_path_to_hip_runtime_dylib()
        # Escape backslashes for C string embedding
        libhip_path_escaped = libhip_path.replace('\\', '\\\\')
        src = Path(os.path.join(dirname, "driver.c")).read_text()
        src = src.replace('/*py_libhip_search_path*/', libhip_path_escaped, 1)
        mod = compile_module_from_src(src=src, name="hip_utils", include_dirs=include_dirs)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        # triton-windows style: launch and arg metadata live in hip_utils, not
        # in per-kernel .pyd files.  This eliminates the Windows crash that
        # occurred when per-kernel C modules tried to call hipGetProcAddress.
        self.launch = mod.launch
        self.build_signature_metadata = mod.build_signature_metadata
        global PyTDMDescriptor, PyKernelArg, ARG_CONSTEXPR, ARG_KERNEL, ARG_TUPLE
        PyTDMDescriptor = mod.PyTDMDescriptor
        PyKernelArg = mod.PyKernelArg
        ARG_CONSTEXPR = mod.ARG_CONSTEXPR
        ARG_KERNEL = mod.ARG_KERNEL
        ARG_TUPLE = mod.ARG_TUPLE


# -------------------- Launcher ----------------------------
def ty_to_cpp(ty):
    if ty[0] == '*':
        return "hipDeviceptr_t"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
    }[ty]


FLOAT_STORAGE_TYPE = {
    "fp16": "uint16_t",
    "bf16": "uint16_t",
    "fp32": "uint32_t",
    "f32": "uint32_t",
    "fp64": "uint64_t",
}
FLOAT_PACK_FUNCTION = {
    "fp16": "pack_fp16",
    "bf16": "pack_bf16",
    "fp32": "pack_fp32",
    "f32": "pack_fp32",
    "fp64": "pack_fp64",
}

_BASE_ARGS_FORMAT = "piiiKKOOOO"


def make_launcher(constants, signature, warp_size):

    def _expand_signature(signature):
        output = []
        # Expand tensor descriptor arguments into base pointer, shape, and
        # strides
        for sig in signature:
            if isinstance(sig, str) and sig.startswith("tensordesc"):
                ndim = sig.count(",") + 1
                dtype = re.match("tensordesc<([^[>]*)", sig).group()

                output.append("*" + dtype)
                for _ in range(2 * ndim):
                    output.append("i64")
                # Currently the host side tensor descriptors get passed in as a
                # tensor desc, shape, and strides. We have no way to use these
                # shape and strides when processing tensor descriptors which is
                # why we provide our own decomposition above. Sadly this means
                # we have to pass the shape and strides twice.
                for _ in range(ndim):
                    output.append("i32")
                for _ in range(ndim):
                    output.append("i64")
            else:
                output.append(sig)

        return output

    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty == "constexpr":
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty == "constexpr":
            return "O"
        return {
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    signature = {idx: s for idx, s in enumerate(_expand_signature(signature.values()))}

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    format = _BASE_ARGS_FORMAT + args_format
    signature = ','.join(map(_serialize_signature, signature.values()))
    signature = list(filter(bool, signature.split(',')))
    signature = {i: s for i, s in enumerate(signature)}
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    arg_decl_list = []
    for i, ty in signature.items():
        if ty == "constexpr":
            continue
        if ty in FLOAT_STORAGE_TYPE:
            arg_decl_list.append(f"{FLOAT_STORAGE_TYPE[ty]} arg{i}")
        else:
            arg_decl_list.append(f"{ty_to_cpp(ty)} arg{i}")
    arg_decls = ', '.join(arg_decl_list)
    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty in FLOAT_STORAGE_TYPE:
            internal_args_list.append(f"_arg{i}_storage")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")

    float_storage_decls = [
        f"{FLOAT_STORAGE_TYPE[ty]} _arg{i}_storage = {FLOAT_PACK_FUNCTION[ty]}(_arg{i});"
        for i, ty in signature.items()
        if ty in FLOAT_STORAGE_TYPE
    ]

    libhip_path = _get_path_to_hip_runtime_dylib()

    # generate glue code
    params = list(range(len(signature)))
    params = [f"&arg{i}" for i, ty in signature.items() if ty != "constexpr"]
    params.append("&global_scratch")
    src = f"""
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <Python.h>
//#include <dlfcn.h>
#include <stdbool.h>
#include <Windows.h>

// The list of paths to search for the HIP runtime library. The caller Python
// code should substitute the search path placeholder.
static const char *hipLibSearchPaths[] = {{"{libhip_path}"}};

// The list of HIP dynamic library symbols and their signature we are interested
// in this file.
#define HIP_SYMBOL_LIST(FOR_EACH_ERR_FN, FOR_EACH_STR_FN)                     \\
  FOR_EACH_STR_FN(hipGetErrorString, hipError_t hipError)                     \\
  FOR_EACH_ERR_FN(hipModuleLaunchKernel, hipFunction_t f,                     \\
                  unsigned int gridDimX, unsigned int gridDimY,               \\
                  unsigned int gridDimZ, unsigned int blockDimX,              \\
                  unsigned int blockDimY, unsigned int blockDimZ,             \\
                  unsigned int sharedMemBytes, hipStream_t stream,            \\
                  void **kernelParams, void **extra)                          \\
  FOR_EACH_ERR_FN(hipModuleLaunchCooperativeKernel, hipFunction_t f,          \\
                  unsigned int gridDimX, unsigned int gridDimY,               \\
                  unsigned int gridDimZ, unsigned int blockDimX,              \\
                  unsigned int blockDimY, unsigned int blockDimZ,             \\
                  unsigned int sharedMemBytes, hipStream_t stream,            \\
                  void **kernelParams, void **extra)                          \\
  FOR_EACH_ERR_FN(hipPointerGetAttribute, void *data,                         \\
                  hipPointer_attribute attribute, hipDeviceptr_t ptr)

// The HIP symbol table for holding resolved dynamic library symbols.
struct HIPSymbolTable {{
#define DEFINE_EACH_ERR_FIELD(hipSymbolName, ...)                             \\
  hipError_t (*hipSymbolName)(__VA_ARGS__);
#define DEFINE_EACH_STR_FIELD(hipSymbolName, ...)                             \\
  const char *(*hipSymbolName)(__VA_ARGS__);

  HIP_SYMBOL_LIST(DEFINE_EACH_ERR_FIELD, DEFINE_EACH_STR_FIELD)
}};

static struct HIPSymbolTable hipSymbolTable;

bool initSymbolTable() {{
  void *lib = NULL;

  // Load the HIP runtime DLL using plain LoadLibraryA - the same approach
  // as driver.c (hip_utils). The absolute path is injected at compile time
  // so DLL search order doesn't matter.
  int n = sizeof(hipLibSearchPaths) / sizeof(hipLibSearchPaths[0]);
  for (int i = 0; i < n; ++i) {{
    HMODULE handle = LoadLibraryA(hipLibSearchPaths[i]);
    if (handle) {{
      lib = (void*)handle;
    }}
  }}
  if (!lib) {{
    PyErr_SetString(PyExc_RuntimeError, "cannot open amdhip64.dll");
    return false;
  }}

  // Resolve all symbols directly via GetProcAddress.
  // We deliberately avoid hipGetProcAddress because it triggers a crash
  // (0xC0000005 at amdhip64_7.dll+0x3af99d) when called from a per-kernel
  // .pyd context. All required symbols are directly exported by amdhip64.dll
  // so GetProcAddress is sufficient and does not require version information.
#define QUERY_EACH_FN(hipSymbolName, ...)                                     \
  *(void **)&hipSymbolTable.hipSymbolName =                                   \
      (void*)GetProcAddress((HMODULE)lib, #hipSymbolName);                    \
  if (!hipSymbolTable.hipSymbolName) {{                                        \
    PyErr_SetString(PyExc_RuntimeError,                                       \
                    "cannot get address for '" #hipSymbolName                 \
                    "' from amdhip64.dll");                                   \
    FreeLibrary((HMODULE)lib);                                                \
    return false;                                                             \
  }}

  HIP_SYMBOL_LIST(QUERY_EACH_FN, QUERY_EACH_FN)

  return true;
}}

static inline void gpuAssert(hipError_t code, const char *file, int line)
{{
   if (code != HIP_SUCCESS)
   {{
      const char* prefix = "Triton Error [HIP]: ";
       const char* str = hipSymbolTable.hipGetErrorString(code);
      char err[1024] = {{0}};
      snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str );
      PyErr_SetString(PyExc_RuntimeError, err);
   }}
}}

#define HIP_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int launch_cooperative_grid, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, hipStream_t stream, hipFunction_t function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  hipDeviceptr_t global_scratch = 0;
  void *params[] = {{ {', '.join(params)} }};
  if (gridX*gridY*gridZ > 0 && launch_cooperative_grid) {{
    HIP_CHECK(hipSymbolTable.hipModuleLaunchCooperativeKernel(function, gridX, gridY, gridZ, {warp_size}*num_warps, 1, 1, shared_memory, stream, params, 0));
    return;
  }}
  if (gridX*gridY*gridZ > 0) {{
    HIP_CHECK(hipSymbolTable.hipModuleLaunchKernel(function, gridX, gridY, gridZ, {warp_size}*num_warps, 1, 1, shared_memory, stream, params, 0));
  }}
}}

typedef struct _DevicePtrInfo {{
    hipDeviceptr_t dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    hipError_t status = hipSymbolTable.hipPointerGetAttribute(&dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == hipErrorInvalidValue) {{
      PyErr_Format(PyExc_ValueError,
                    "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
      ptr_info.valid = false;
    }}
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static uint16_t pack_fp16(double f) {{
    uint16_t result;
    // from https://github.com/python/pythoncapi-compat
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
    _PyFloat_Pack2(f, (char*)&result, 1);
#else
    PyFloat_Pack2(f, (char*)&result, 1);
#endif
    return result;
}}

static uint16_t pack_bf16(double f) {{
    float f32 = (float)f;
    uint32_t u32 = *(uint32_t*)&f32;
    return (uint16_t)(u32 >> 16);
}}

static uint32_t pack_fp32(double f) {{
    float f32 = (float)f;
    return *(uint32_t*)&f32;
}}

static uint64_t pack_fp64(double f) {{
    return *(uint64_t*)&f;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int launch_cooperative_grid;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &launch_cooperative_grid,
                                           &gridX, &gridY, &gridZ, &_stream, &_function,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  {' '.join(float_storage_decls)}

  // extract kernel metadata
  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
    return NULL;
  }}
  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, launch_cooperative_grid, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  if(PyErr_Occurred()) {{
    return NULL;
  }}
  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  if (!initSymbolTable()) {{
    return NULL;
  }}
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


# -------------------- triton-windows-style launch helpers --------------------
# These replace the per-kernel C generation (make_launcher) that crashed on
# Windows.  Instead, a single shared launch function in driver.c / hip_utils
# handles all kernels generically via argument annotations and signature bytes.

def _expand_signature_for_launch(signature, tensordesc_meta=None):
    """Expand tensordesc signatures to match triton-tiger's _expand_signature."""
    output = []
    for sig in signature:
        if isinstance(sig, str) and sig.startswith("tensordesc"):
            ndim = sig.count(",") + 1
            match = re.match(r"tensordesc<([^[>]*)", sig)
            dtype = match.group(1) if match else "fp32"
            output.append("*" + dtype)
            for _ in range(2 * ndim):
                output.append("i64")
            # Shape and strides passed twice (host-side decomposition)
            for _ in range(ndim):
                output.append("i32")
            for _ in range(ndim):
                output.append("i64")
        else:
            output.append(sig)
    return output


def _flatten_signature(sig, output):
    if isinstance(sig, tuple):
        for x in sig:
            _flatten_signature(x, output)
    else:
        output.append(sig)


def _make_kernel_signature(expanded_signature):
    """Return bytes encoding the extractor index for each non-constexpr arg."""
    import triton
    flat = []
    for sig in expanded_signature:
        _flatten_signature(sig, flat)
    kernel_sig = [x for x in flat if x != "constexpr"]
    return triton.runtime.driver.active.utils.build_signature_metadata(kernel_sig)


def _annotate_arguments(signature):
    """Build PyKernelArg annotation list for the generic launch function."""
    annotated = []
    for sig in signature:
        if isinstance(sig, tuple):
            annotated.append(PyKernelArg(nested_tuple=_annotate_arguments(sig), type=ARG_TUPLE))
        elif sig != "constexpr":
            annotated.append(PyKernelArg(nested_tuple=None, type=ARG_KERNEL))
        else:
            annotated.append(PyKernelArg(nested_tuple=None, type=ARG_CONSTEXPR))
    return annotated


class HIPLauncher(object):
    """
    Kernel launcher that delegates to the shared launch function in hip_utils
    (driver.c) instead of generating a per-kernel C .pyd.  This matches the
    triton-windows architecture and avoids the 0xC0000005 crash caused by
    per-kernel modules calling hipGetProcAddress at init time.
    """

    def __init__(self, src, metadata):
        import triton
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x),) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}

        expanded = _expand_signature_for_launch(list(signature.values()))
        self.arg_annotations = _annotate_arguments(expanded)
        self.kernel_signature = _make_kernel_signature(expanded)
        self.launch = triton.runtime.driver.active.utils.launch
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.warp_size = metadata.warp_size

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata,
                 launch_metadata, launch_enter_hook, launch_exit_hook, *args):
        # kernel_metadata = (num_warps, num_ctas, shared, clusterX, clusterY, clusterZ)
        # Matches the "piiiKK(iiiiii)OOOiOy*O" format in driver.c launchKernel.
        self.launch(self.launch_cooperative_grid, gridX, gridY, gridZ,
                    stream, function, kernel_metadata, launch_metadata,
                    launch_enter_hook, launch_exit_hook,
                    self.warp_size, self.arg_annotations,
                    self.kernel_signature, args)


class HIPDriver(GPUDriver):

    def __init__(self):
        super().__init__()
        self.utils = HIPUtils()
        self.launcher_cls = HIPLauncher

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        import torch
        if not torch.cuda.is_available():
            return False
        if torch.version.hip is not None:
            return True
        cc = torch.cuda.get_device_capability()
        if cc == (8, 8):
            return True
        return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_current_target(self):
        device = self.get_current_device()
        device_properties = self.utils.get_device_properties(device)
        arch = knobs.runtime.override_arch or device_properties['arch']
        warp_size = device_properties['warpSize']
        return GPUTarget("hip", arch.split(':')[0], warp_size)

    def get_active_torch_device(self):
        import torch
        # when using hip devices, the device string in pytorch is "cuda"
        return torch.device("cuda", self.get_current_device())

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # It's the same as the Nvidia backend.
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

    def clear_cache(self, cache):
        cache.zero_()
