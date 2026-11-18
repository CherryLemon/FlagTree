from triton import knobs

TLE_SUPPORTED_PRIMITIVES = [
    ## TLE-Raw
    "raw.call",
    "raw.call_smem",

    ## TLE-Lite
    "extract_tile",
    "cumsum",

    ## TLE-Struct GPU
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.set_layout",
]

# The HCU compiler verifies the supported subset before lowering these opt-ins.
if knobs.hcu.experimental_software_ws:
    TLE_SUPPORTED_PRIMITIVES.append("gpu.warp_specialize")

if knobs.hcu.experimental_static_pipe:
    TLE_SUPPORTED_PRIMITIVES.append("pipe")
