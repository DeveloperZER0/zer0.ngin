// Thin Win32 layer: timing, file mapping, affinity. Avoids std.Io churn.
const std = @import("std");
pub extern "kernel32" fn QueryPerformanceCounter(*i64) callconv(.winapi) i32;
pub extern "kernel32" fn QueryPerformanceFrequency(*i64) callconv(.winapi) i32;
pub extern "kernel32" fn LoadLibraryA([*:0]const u8) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn GetProcAddress(*anyopaque, [*:0]const u8) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn GetLastError() callconv(.winapi) u32;
pub extern "kernel32" fn VirtualAlloc(?*anyopaque, usize, u32, u32) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn SetThreadAffinityMask(*anyopaque, usize) callconv(.winapi) usize;
pub extern "kernel32" fn SetThreadPriority(*anyopaque, i32) callconv(.winapi) i32;
pub extern "kernel32" fn GetCurrentThread() callconv(.winapi) *anyopaque;
pub extern "kernel32" fn Sleep(u32) callconv(.winapi) void;
pub extern "kernel32" fn SetConsoleOutputCP(u32) callconv(.winapi) i32;
pub extern "kernel32" fn GetCommandLineA() callconv(.winapi) [*:0]const u8;
pub extern "kernel32" fn ExitProcess(u32) callconv(.winapi) noreturn;
extern "kernel32" fn CreateFileA([*:0]const u8, u32, u32, ?*anyopaque, u32, u32, ?*anyopaque) callconv(.winapi) *anyopaque;
extern "kernel32" fn GetFileSizeEx(*anyopaque, *i64) callconv(.winapi) i32;
extern "kernel32" fn CreateFileMappingA(*anyopaque, ?*anyopaque, u32, u32, u32, ?[*:0]const u8) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn MapViewOfFile(*anyopaque, u32, u32, u32, usize) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn ReadFile(*anyopaque, [*]u8, u32, *u32, ?*anyopaque) callconv(.winapi) i32;
extern "kernel32" fn WriteFile(*anyopaque, [*]const u8, u32, *u32, ?*anyopaque) callconv(.winapi) i32;
extern "kernel32" fn CloseHandle(*anyopaque) callconv(.winapi) i32;
extern "kernel32" fn GetStdHandle(u32) callconv(.winapi) *anyopaque;
const INVALID: usize = std.math.maxInt(usize);

pub fn now() f64 {
    var a: i64 = 0;
    var f: i64 = 0;
    _ = QueryPerformanceCounter(&a);
    _ = QueryPerformanceFrequency(&f);
    return @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(f));
}
pub fn mapFile(path: [:0]const u8) ![]const u8 {
    const h = CreateFileA(path, 0x80000000, 1, null, 3, 0x80, null);
    if (@intFromPtr(h) == INVALID) return error.OpenFailed;
    var sz: i64 = 0;
    if (GetFileSizeEx(h, &sz) == 0) return error.SizeFailed;
    const m = CreateFileMappingA(h, null, 2, 0, 0, null) orelse return error.MapFailed;
    const p = MapViewOfFile(m, 4, 0, 0, 0) orelse return error.ViewFailed;
    return @as([*]const u8, @ptrCast(p))[0..@intCast(sz)];
}
pub fn readFile(alloc: std.mem.Allocator, path: [:0]const u8) ![]u8 {
    const h = CreateFileA(path, 0x80000000, 1, null, 3, 0x80, null);
    if (@intFromPtr(h) == INVALID) return error.OpenFailed;
    defer _ = CloseHandle(h);
    var sz: i64 = 0;
    if (GetFileSizeEx(h, &sz) == 0) return error.SizeFailed;
    const b = try alloc.alloc(u8, @intCast(sz));
    var o: usize = 0;
    while (o < b.len) {
        var n: u32 = 0;
        if (ReadFile(h, b.ptr + o, @intCast(@min(b.len - o, 1 << 30)), &n, null) == 0 or n == 0) return error.ReadFailed;
        o += n;
    }
    return b;
}
pub fn writeFile(path: [:0]const u8, data: []const u8) !void {
    const h = CreateFileA(path, 0x40000000, 0, null, 2, 0x80, null);
    if (@intFromPtr(h) == INVALID) return error.OpenFailed;
    defer _ = CloseHandle(h);
    var o: usize = 0;
    while (o < data.len) {
        var n: u32 = 0;
        if (WriteFile(h, data.ptr + o, @intCast(@min(data.len - o, 1 << 30)), &n, null) == 0) return error.WriteFailed;
        o += n;
    }
}
pub fn out(bytes: []const u8) void {
    var n: u32 = 0;
    _ = WriteFile(GetStdHandle(0xFFFFFFF5), bytes.ptr, @intCast(bytes.len), &n, null);
}
pub fn pages(n: usize) ![]align(64) u8 {
    const p = VirtualAlloc(null, n, 0x3000, 4) orelse return error.OutOfMemory;
    return @as([*]align(64) u8, @ptrCast(@alignCast(p)))[0..n];
}
