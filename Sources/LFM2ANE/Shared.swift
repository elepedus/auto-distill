// Memory shared between MLX (GPU) and ANE programs without copies.
import ANE
import Foundation
import IOSurface
import MLX

/// Zero-copy MLX view of an ANE IOSurface binding, honouring the compiler's strides.
public func view(_ b: SurfaceBinding) -> MLXArray {
    let flat = MLXArray(rawPointer: IOSurfaceGetBaseAddress(b.surface), [IOSurfaceGetAllocSize(b.surface) / 2], dtype: .float16) { _ = b }
    let l = b.layout
    return asStrided(flat, [l.shape.batches, l.shape.channels, l.shape.height, l.shape.width],
                     strides: [l.batchStride / 2, l.planeStride / 2, l.rowStride / 2, 1], offset: b.startOffset / 2)
}

public func tensor(_ a: MLXArray, _ dims: [Int]) throws -> Tensor {
    try Tensor(shape: TensorShape(batches: dims[0], channels: dims[1], height: dims[2], width: dims[3]),
               values: a.asType(.float16).asArray(Float16.self))
}

/// Page-aligned fp16 memory we own. ANE programs bind it as an input; MLX reads it in place through
/// `array` and writes it in place with `write`. One buffer serves every program and the GPU.
public final class SharedBuffer: @unchecked Sendable {
    public let pointer: UnsafeMutableRawPointer, bytes: Int, shape: [Int]
    public let array: MLXArray  // logical elements, viewed in place

    public init(_ shape: [Int]) {
        let page = Int(getpagesize()), count = shape.reduce(1, *)
        bytes = max(65536, (count * 2 + page - 1) / page * page)  // ANE allocations are at least 64 KiB
        var p: UnsafeMutableRawPointer?
        precondition(posix_memalign(&p, page, bytes) == 0)
        memset(p!, 0, bytes)
        pointer = p!; self.shape = shape
        // ponytail: never freed; buffers live for the whole process.
        array = MLXArray(rawPointer: pointer, [bytes / 2], dtype: .float16) {}[0 ..< count]
    }

    public func binding() throws -> ExternalInputBinding {
        try .callerMemory(pointer: pointer, allocationBytes: bytes, byteRange: 0 ..< bytes,
                          layout: .tightlyPacked(shape: TensorShape(batches: shape[0], channels: shape[1], height: shape[2], width: shape[3]), dataType: .float16),
                          retaining: self, access: .readOnly)
    }

    /// Writes `values` (any shape with the same element count) on the GPU. Lazy: returns a marker to evaluate.
    public func write(_ values: MLXArray) -> MLXArray {
        let n = array.size
        precondition(values.size == n)
        return writeKernel([values.reshaped(-1), array], grid: (n, 1, 1), threadGroup: (256, 1, 1),
                           outputShapes: [[1]], outputDTypes: [.int32])[0]
    }
}

// MLX kernels take inputs as const, so writing into a shared buffer casts the const away. MLX never sees the
// write; callers evaluate the returned marker before an ANE program reads the buffer.
private let writeKernel = MLXFast.metalKernel(
    name: "write_into_shared", inputNames: ["src", "dst"], outputNames: ["done"],
    source: """
        uint i = thread_position_in_grid.x;
        if (i < src_shape[0]) ((device half*)dst)[i] = half(src[i]);
        if (i == 0) done[0] = 1;
        """)

/// One TTF step for P members' [N, K] ternary matrices, in place: leaky accumulation of the Adam-normalised
/// step, then a trit moves one level where the accumulator reaches theta, which is spent. Per member p,
/// params[4p ..< 4p+4] = [rho, leak, theta, mean(row moment)]. Template args N, K. Outputs the new accumulator
/// and a 0/1 map of moved trits; the trits are rewritten in the shared buffer.
public let ttfKernel = MLXFast.metalKernel(
    name: "ttf_fire", inputNames: ["g", "acc", "rowv", "colv", "params", "trits"], outputNames: ["acc_out", "moved"],
    source: """
        uint idx = thread_position_in_grid.x;
        if (idx >= g_shape[0]) return;
        uint p = idx / (N * K), local = idx % (N * K), i = local / K, j = local % K;
        device const float* q = params + 4 * p;
        float v = rowv[p * N + i] * colv[p * K + j] / q[3];
        float a = q[1] * acc[idx] - q[0] * g[idx] / (metal::sqrt(v) + 1e-20f);
        float t = float(trits[idx]), d = 0.0f;
        if (a >= q[2] && t < 1.0f) d = 1.0f;
        else if (a <= -q[2] && t > -1.0f) d = -1.0f;
        if (d != 0.0f) ((device half*)trits)[idx] = half(t + d);
        acc_out[idx] = a - q[2] * d;
        moved[idx] = d != 0.0f ? 1.0f : 0.0f;
        """)
