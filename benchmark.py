import pyopencl as cl
import numpy as np
import time

# Setup
platforms = cl.get_platforms()
devices = platforms[0].get_devices()
ctx = cl.Context(devices)
queue = cl.CommandQueue(ctx)


# Optimized kernel - vectorized FMA with unrolling
kernel_code = """
__kernel void compute(__global float4* data) {
    int gid = get_global_id(0);
    float4 v1 = data[gid];
    float4 v2 = v1 * 1.1f;
    float4 v3 = v1 * 1.2f;
    float4 v4 = v1 * 1.3f;
    float4 v5 = v1 * 1.4f;
    float4 v6 = v1 * 1.5f;
    float4 v7 = v1 * 1.6f;
    float4 v8 = v1 * 1.7f;
     
    for(int i = 0; i < (1<<20); i++) {
         v1 = v1 * 1.0001f + 0.0001f;
         v2 = v2 * 1.0001f + 0.0001f;
         v3 = v3 * 1.0001f + 0.0001f;
         v4 = v4 * 1.0001f + 0.0001f;
         v5 = v5 * 1.0001f + 0.0001f;
         v6 = v6 * 1.0001f + 0.0001f;
         v7 = v7 * 1.0001f + 0.0001f;
         v8 = v8 * 1.0001f + 0.0001f;
    }
     
    data[gid] = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8;
}
"""
    
prg = cl.Program(ctx, kernel_code).build()
kernel = cl.Kernel(prg, "compute")

# Data - increased size
n = 1 << 16 # 65536 threads
data = np.zeros(n * 4, dtype=np.float32)
# print(data)
buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)

# Warmup
# kernel(queue, (n,), None, buf)
queue.finish()

# Benchmark
start = time.time()
iterations = 0

while time.time() - start < 1:
    # print(n)
    # print(time.time() - start)
    kernel(queue, (n,), None, buf).wait()
    iterations += 1

queue.finish()
elapsed = time.time() - start

# Read buffer back
result = np.empty_like(data)
cl.enqueue_copy(queue, result, buf).wait()
np.set_printoptions(threshold=np.inf)
# print(f"Buffer contents: {result}")
print(f"Sum of all elements: {result.sum()}")

flops_per_kernel = n * (1<<20) * 4 * 8 * 2  # n threads * iterations * 4 elements * 8 vectors * 2 ops (mul+add)
total_flops = flops_per_kernel * iterations
tflops = (total_flops / elapsed) / 1e12

print(f"Device: {devices[0].name}")
print(f"Threads: {n}")
print(f"Iterations: {iterations}")
print(f"Time: {elapsed:.3f}s")
print(f"TFLOPS: {tflops:.3f}")
print(f"Total TFLOPs: {total_flops / 1e12}")
