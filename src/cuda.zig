// CUDA driver API + NVRTC loaded at run time: no CUDA toolkit, no import libraries.
const std = @import("std");
const win = @import("win.zig");
pub const P = ?*anyopaque;
pub const D = u64;

pub var cuInit: *const fn (c_uint) callconv(.c) c_int = undefined;
pub var cuDeviceGet: *const fn (*c_int, c_int) callconv(.c) c_int = undefined;
pub var cuCtxCreate_v2: *const fn (*P, c_uint, c_int) callconv(.c) c_int = undefined;
pub var cuCtxSetCurrent: *const fn (P) callconv(.c) c_int = undefined;
pub var cuModuleLoadData: *const fn (*P, *const anyopaque) callconv(.c) c_int = undefined;
pub var cuModuleGetFunction: *const fn (*P, P, [*:0]const u8) callconv(.c) c_int = undefined;
pub var cuMemHostAlloc: *const fn (*P, usize, c_uint) callconv(.c) c_int = undefined;
pub var cuMemHostGetDevicePointer_v2: *const fn (*D, P, c_uint) callconv(.c) c_int = undefined;
pub var cuMemAlloc_v2: *const fn (*D, usize) callconv(.c) c_int = undefined;
pub var cuMemFree_v2: *const fn (D) callconv(.c) c_int = undefined;
pub var cuMemcpyHtoD_v2: *const fn (D, *const anyopaque, usize) callconv(.c) c_int = undefined;
pub var cuMemcpyHtoDAsync_v2: *const fn (D, *const anyopaque, usize, P) callconv(.c) c_int = undefined;
pub var cuMemcpyDtoDAsync_v2: *const fn (D, D, usize, P) callconv(.c) c_int = undefined;
pub var cuMemcpyDtoH_v2: *const fn (*anyopaque, D, usize) callconv(.c) c_int = undefined;
pub var cuMemcpyDtoHAsync_v2: *const fn (*anyopaque, D, usize, P) callconv(.c) c_int = undefined;
pub var cuMemsetD8_v2: *const fn (D, u8, usize) callconv(.c) c_int = undefined;
pub var cuMemsetD8Async: *const fn (D, u8, usize, P) callconv(.c) c_int = undefined;
pub var cuFuncGetAttribute: *const fn (*c_int, c_int, P) callconv(.c) c_int = undefined;
pub var cuStreamCreate: *const fn (*P, c_uint) callconv(.c) c_int = undefined;
pub var cuStreamSynchronize: *const fn (P) callconv(.c) c_int = undefined;
pub var cuStreamQuery: *const fn (P) callconv(.c) c_int = undefined;
pub var cuStreamWaitEvent: *const fn (P, P, c_uint) callconv(.c) c_int = undefined;
pub var cuEventCreate: *const fn (*P, c_uint) callconv(.c) c_int = undefined;
pub var cuEventRecord: *const fn (P, P) callconv(.c) c_int = undefined;
pub var cuEventQuery: *const fn (P) callconv(.c) c_int = undefined;
pub var cuLaunchKernel: *const fn (P, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, P, ?[*]const ?*const anyopaque, P) callconv(.c) c_int = undefined;
pub var cuMemGetInfo_v2: *const fn (*usize, *usize) callconv(.c) c_int = undefined;

fn sym(h: *anyopaque, comptime T: type, n: [*:0]const u8) T {
    return @ptrCast(@alignCast(win.GetProcAddress(h, n) orelse std.debug.panic("missing {s}", .{n})));
}
pub fn ck(r: c_int) void {
    if (r != 0) std.debug.panic("CUDA error {d}", .{r});
}
pub var stream: P = null;
pub var copy_stream: P = null;
pub var ctx: P = null;

pub fn init() !void {
    const h = win.LoadLibraryA("nvcuda.dll") orelse return error.NoCuda;
    inline for (.{
        "cuInit", "cuDeviceGet", "cuCtxCreate_v2", "cuCtxSetCurrent", "cuModuleLoadData", "cuModuleGetFunction", "cuMemHostAlloc",
        "cuMemHostGetDevicePointer_v2", "cuMemAlloc_v2", "cuMemFree_v2", "cuMemcpyHtoD_v2", "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoDAsync_v2", "cuMemcpyDtoH_v2",
        "cuMemcpyDtoHAsync_v2", "cuMemsetD8_v2", "cuMemsetD8Async", "cuFuncGetAttribute", "cuStreamCreate", "cuStreamSynchronize", "cuStreamQuery", "cuStreamWaitEvent", "cuEventCreate", "cuEventRecord",
        "cuEventQuery", "cuLaunchKernel", "cuMemGetInfo_v2",
    }) |n| @field(@This(), n) = sym(h, @TypeOf(@field(@This(), n)), n);
    ck(cuInit(0));
    var dev: c_int = 0;
    ck(cuDeviceGet(&dev, 0));
    ck(cuCtxCreate_v2(&ctx, 0x8 | 0x1, dev)); // spin, map host
    ck(cuStreamCreate(&stream, 0));
    ck(cuStreamCreate(&copy_stream, 0));
}
pub fn alloc(n: usize) D {
    var d: D = 0;
    const r = cuMemAlloc_v2(&d, n);
    if (r != 0) std.debug.panic("VRAM allocation of {d} MiB failed ({d})", .{ n >> 20, r });
    ck(cuMemsetD8_v2(d, 0, n));
    return d;
}
pub const Host = struct { p: [*]align(64) u8, d: D };
pub fn hostAlloc(n: usize, flags: c_uint) !Host {
    var p: P = null;
    if (cuMemHostAlloc(&p, n, flags) != 0) return error.PinFailed;
    var d: D = 0;
    if (flags & 2 != 0) ck(cuMemHostGetDevicePointer_v2(&d, p, 0));
    return .{ .p = @ptrCast(@alignCast(p.?)), .d = d };
}
pub fn upload(bytes: []const u8) D {
    const d = alloc(std.mem.alignForward(usize, bytes.len, 256));
    ck(cuMemcpyHtoD_v2(d, bytes.ptr, bytes.len));
    return d;
}
pub var prof: bool = false;
pub const Prof = struct { f: P, t: f64, n: u64 };
pub var profs: [64]Prof = @splat(.{ .f = null, .t = 0, .n = 0 });
/// Launch on the main stream. args is a tuple of D / i32 / u32 values.
pub inline fn launch(f: P, gx: u32, gy: u32, bx: u32, args: anytype) void {
    const fields = @typeInfo(@TypeOf(args)).@"struct".fields;
    var vals: [fields.len]u64 align(8) = undefined; // little endian: the low bytes hold 4-byte arguments
    var ptrs: [fields.len]?*const anyopaque = undefined;
    inline for (fields, 0..) |fl, i| {
        vals[i] = switch (fl.type) {
            u64, u32 => @field(args, fl.name),
            i32 => @as(u32, @bitCast(@field(args, fl.name))),
            else => 0,
        };
        // structs are passed by reference to the caller's copy
        ptrs[i] = if (@typeInfo(fl.type) == .pointer) @ptrCast(@field(args, fl.name)) else @ptrCast(&vals[i]);
    }
    if (prof) _ = cuStreamSynchronize(stream);
    const t0 = if (prof) win.now() else 0;
    const r = cuLaunchKernel(f, gx, gy, 1, bx, 1, 1, 0, stream, &ptrs, null);
    if (prof) {
        _ = cuStreamSynchronize(stream);
        for (&profs) |*e| if (e.f == f or e.f == null) {
            e.f = f;
            e.t += win.now() - t0;
            e.n += 1;
            break;
        };
    }
    if (r != 0) std.debug.panic("launch failed {d}", .{r});
}

/// Compile with NVRTC, or load the cubin cached for this exact source.
pub fn module(a: std.mem.Allocator, src: [:0]const u8, dir: []const u8) !P {
    const hash = std.hash.Wyhash.hash(7, src);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/kernels-{x}.cubin", .{ dir, hash }, 0);
    var m: P = null;
    if (win.readFile(a, path)) |bin| {
        if (cuModuleLoadData(&m, bin.ptr) == 0) return m;
    } else |_| {}
    const dll = try std.fmt.allocPrintSentinel(a, "{s}/nvrtc64_120_0.dll", .{dir}, 0);
    const h = win.LoadLibraryA(dll) orelse win.LoadLibraryA("nvrtc64_120_0.dll") orelse return error.NoNvrtc;
    const create = sym(h, *const fn (*P, [*:0]const u8, ?[*:0]const u8, c_int, P, P) callconv(.c) c_int, "nvrtcCreateProgram");
    const compile = sym(h, *const fn (P, c_int, [*]const [*:0]const u8) callconv(.c) c_int, "nvrtcCompileProgram");
    const size = sym(h, *const fn (P, *usize) callconv(.c) c_int, "nvrtcGetCUBINSize");
    const get = sym(h, *const fn (P, [*]u8) callconv(.c) c_int, "nvrtcGetCUBIN");
    const lsize = sym(h, *const fn (P, *usize) callconv(.c) c_int, "nvrtcGetProgramLogSize");
    const lget = sym(h, *const fn (P, [*]u8) callconv(.c) c_int, "nvrtcGetProgramLog");
    var p: P = null;
    ck(create(&p, src, "kernels.cu", 0, null, null));
    const opts = [_][*:0]const u8{ "-arch=sm_86", "-std=c++17" };
    const r = compile(p, opts.len, &opts);
    if (r != 0) {
        var n: usize = 0;
        _ = lsize(p, &n);
        const log = try a.alloc(u8, n);
        _ = lget(p, log.ptr);
        std.debug.print("{s}\n", .{log});
        return error.CompileFailed;
    }
    var n: usize = 0;
    ck(size(p, &n));
    const bin = try a.alloc(u8, n);
    ck(get(p, bin.ptr));
    win.writeFile(path, bin) catch {};
    ck(cuModuleLoadData(&m, bin.ptr));
    return m;
}
