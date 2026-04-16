# RDNA2 / Windows Fix — lshqqytiger/triton

Patches for running https://github.com/lshqqytiger/triton on **AMD RDNA2 (gfx1030, RX 6800)** under
**Windows** with ROCm SDK 7.1 / HIP 7.2.  Upstream lshqqytiger/triton targets Linux
and newer HIP runtimes; these files correct four areas that crash or fail to
compile on this platform.

---

## Affected files

```
rdna2-fix/
  backends/amd/driver.c      ← C module compiled once as hip_utils
  backends/amd/driver.py     ← Python AMD backend driver
  backends/amd/compiler.py   ← AMD kernel compiler
  runtime/build.py           ← Module build / clang invocation
```

Drop these files over the corresponding paths inside the triton installation
(e.g. `<venv>/Lib/site-packages/triton/`) and clear the Triton cache
(`~/.triton/cache`).

---

## Problem 1 — 0xC0000005 crash in `PyInit___triton_launcher`

**File:** `backends/amd/driver.py` + `backends/amd/driver.c`

### Root cause
Upstream triton-tiger generates a **per-kernel** C `.pyd` file
(`__triton_launcher.cp312-win_amd64.pyd`) for every compiled kernel.
Each file calls `hipGetProcAddress` inside `PyInit___triton_launcher`.
On Windows with ROCm 7.1, this consistently crashes at
`amdhip64_7.dll+0x3af99d` regardless of load flags — the HIP
version-aware resolver is not usable at DLL init time on Windows.

### Fix
Ported the **triton-windows** architecture:

* `driver.c` is compiled **once** as a shared `hip_utils` module.  It uses
  `GetProcAddress` (via a `dlsym`→`GetProcAddress` shim) to resolve HIP
  symbols — no `hipGetProcAddress`, no per-kernel init crashes.
* `HIPLauncher` in `driver.py` no longer generates per-kernel C code.
  It instead calls `triton.runtime.driver.active.utils.launch` with
  argument-annotation metadata (`build_signature_metadata` / `PyKernelArg`)
  that was already present in the `hip_utils` C module.

### Targeted adaptations for lshqqytiger/triton (vs triton-windows)
| Detail | triton-windows | lshqqytiger/triton (this fix) |
|--------|---------------|------------------------|
| `packed_metadata` tuple | 3-element | **6-element** `(num_warps, num_ctas, shared, clusterDimX, clusterDimY, clusterDimZ)` |
| Extra scratch params | `+2` (global + profile) | **`+1`** (global only) |

The `launchKernel` format string in `driver.c` is adjusted accordingly.

---

## Problem 2 — `HIP_LAUNCH_CONFIG` / `hipLaunchAttribute` compile errors

**File:** `backends/amd/driver.c`

```
error: unknown type name 'HIP_LAUNCH_CONFIG'
error: unknown type name 'hipLaunchAttribute'
```

### Root cause
The `hipDrvLaunchKernelEx` entry in `HIP_SYMBOL_LIST` references
`HIP_LAUNCH_CONFIG` and `hipLaunchAttribute`, which are **not defined** in
the ROCm 7.1 / HIP 7.2 headers bundled with the SDK (`_rocm_sdk_core`).
These types exist only in newer HIP versions targeting multi-CTA cluster
launches, a feature RDNA2 does not support anyway.

### Fix
* Removed `hipDrvLaunchKernelEx` from `HIP_SYMBOL_LIST`.
* Replaced the `num_ctas > 1` launch branch with an immediate
  `PyErr_SetString(RuntimeError, ...)` — cluster launches are architecturally
  unsupported on gfx1030.

---

## Problem 3 — `hipGetLastError` wrong macro category

**File:** `backends/amd/driver.c`

```c
// Before (wrong — returns hipError_t, not const char*)
FOR_EACH_STR_FN(hipGetLastError)

// After (correct)
FOR_EACH_ERR_FN(hipGetLastError)
```

`hipGetLastError` is declared as `hipError_t hipGetLastError(void)` in the
HIP headers.  Using `FOR_EACH_STR_FN` caused the generated symbol-table
wrapper to treat the return value as `const char *`, producing undefined
behaviour.

---

## Problem 4 — Windows temp-file `PermissionError` in compiler

**File:** `backends/amd/compiler.py`

Python's `NamedTemporaryFile` on Windows holds an exclusive lock while the
file is open.  Passing the file path to clang while it is still open raises
`PermissionError`.

### Fix
```python
# Before
with tempfile.NamedTemporaryFile(mode='wb', suffix='.hsaco') as f:
    ...

# After
with tempfile.NamedTemporaryFile(mode='wb', suffix='.hsaco', delete=False) as f:
    tmp_path = f.name
# file is closed here — clang can now open it
```

---

## Problem 5 — `build.py` Windows link / path issues

**File:** `runtime/build.py`

Two issues on Windows:

1. **Missing `python312.lib`**: The venv `libs/` directory does not contain
   `python312.lib` by default when using the ROCm Python distribution.
   `build.py` now adds the system Python `libs/` as a fallback linker search
   path.

2. **`HIP_PATH` / `CC` not set**: The ROCm SDK ships its own clang inside
   `_rocm_sdk_core/lib/llvm/bin/`.  `build.py` now discovers and sets
   `HIP_PATH` and `CC` from the installed `_rocm_sdk_core` package if the
   environment variables are absent, making the build self-contained without
   requiring a system-level ROCm installation.

---
## Required runtime environment variables : 

1. set CC = "SD.Next/venv/Lib/site-packages/_rocm_sdk_core/lib/llvm/bin/clang.exe"
2. set HIP_PATH = "SD.Next/venv/Lib/site-packages/_rocm_sdk_core/"


## Reference implementation
These fixes were derived by comparing triton-tiger against the
[triton-windows](https://github.com/triton-lang/triton) reference build
which already handles the Windows/RDNA2 constraints correctly.

**GPU tested:** AMD Radeon RX 6800 (gfx1030, RDNA2)  
**OS:** Windows 11  
**ROCm SDK:** 7.1 (`_rocm_sdk_core`)  
**HIP:** 7.2  
**Python:** 3.12  
**PyTorch:** 2.10+rocm7.12  
