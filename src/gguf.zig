// Minimal GGUF v3 reader over a read-only file mapping.
const std = @import("std");
const win = @import("win.zig");

pub const Type = enum(u32) { f32 = 0, q8_0 = 8, q5_k = 13, bf16 = 30, nvfp4 = 40, _ };
pub const Tensor = struct { name: []const u8, ne: [4]u64, ty: Type, data: [*]const u8 };

pub const Gguf = struct {
    map: []const u8,
    tensors: std.StringHashMap(Tensor),
    kv: std.StringHashMap(usize), // offset of the value's type tag

    pub fn open(alloc: std.mem.Allocator, path: [:0]const u8) !Gguf {
        const m = try win.mapFile(path);
        if (!std.mem.eql(u8, m[0..4], "GGUF")) return error.NotGguf;
        var g: Gguf = .{
            .map = m,
            .tensors = std.StringHashMap(Tensor).init(alloc),
            .kv = std.StringHashMap(usize).init(alloc),
        };
        var r: Rd = .{ .m = m, .o = 8 };
        const nt = r.int(u64);
        const nk = r.int(u64);
        for (0..nk) |_| {
            const k = r.str();
            try g.kv.put(k, r.o);
            const t = r.int(u32);
            r.skip(t);
        }
        const Tmp = struct { name: []const u8, ne: [4]u64, ty: u32, off: u64 };
        const tmp = try alloc.alloc(Tmp, nt);
        defer alloc.free(tmp);
        for (tmp) |*t| {
            t.name = r.str();
            const nd = r.int(u32);
            t.ne = .{ 1, 1, 1, 1 };
            for (0..nd) |i| t.ne[i] = r.int(u64);
            t.ty = r.int(u32);
            t.off = r.int(u64);
        }
        const base = std.mem.alignForward(usize, r.o, 32);
        for (tmp) |t| try g.tensors.put(t.name, .{
            .name = t.name,
            .ne = t.ne,
            .ty = @enumFromInt(t.ty),
            .data = m.ptr + base + t.off,
        });
        return g;
    }
    pub fn get(g: *const Gguf, name: []const u8) !Tensor {
        return g.tensors.get(name) orelse {
            std.debug.print("missing tensor {s}\n", .{name});
            return error.MissingTensor;
        };
    }
    pub fn blk(g: *const Gguf, il: usize, suffix: []const u8) !Tensor {
        var b: [128]u8 = undefined;
        return g.get(try std.fmt.bufPrint(&b, "blk.{d}.{s}", .{ il, suffix }));
    }
    pub fn blkOpt(g: *const Gguf, il: usize, suffix: []const u8) ?Tensor {
        var b: [128]u8 = undefined;
        return g.tensors.get(std.fmt.bufPrint(&b, "blk.{d}.{s}", .{ il, suffix }) catch return null);
    }
    pub fn u32kv(g: *const Gguf, key: []const u8) ?u32 {
        const o = g.kv.get(key) orelse return null;
        var r: Rd = .{ .m = g.map, .o = o };
        _ = r.int(u32);
        return r.int(u32);
    }
    pub fn f32kv(g: *const Gguf, key: []const u8) ?f32 {
        return @bitCast(g.u32kv(key) orelse return null);
    }
    pub fn strArr(g: *const Gguf, alloc: std.mem.Allocator, key: []const u8) ![][]const u8 {
        var r: Rd = .{ .m = g.map, .o = g.kv.get(key) orelse return error.MissingKey };
        if (r.int(u32) != 9 or r.int(u32) != 8) return error.BadType;
        const n = r.int(u64);
        const a = try alloc.alloc([]const u8, n);
        for (a) |*s| s.* = r.str();
        return a;
    }
    pub fn i32Arr(g: *const Gguf, alloc: std.mem.Allocator, key: []const u8) ![]i32 {
        var r: Rd = .{ .m = g.map, .o = g.kv.get(key) orelse return error.MissingKey };
        if (r.int(u32) != 9) return error.BadType;
        const et = r.int(u32);
        const n = r.int(u64);
        const a = try alloc.alloc(i32, n);
        for (a) |*s| s.* = if (et == 7) @as(i32, r.int(u8)) else r.int(i32);
        return a;
    }
};

const Rd = struct {
    m: []const u8,
    o: usize,
    fn int(r: *Rd, comptime T: type) T {
        const v = std.mem.readInt(T, r.m[r.o..][0..@sizeOf(T)], .little);
        r.o += @sizeOf(T);
        return v;
    }
    fn str(r: *Rd) []const u8 {
        const n = r.int(u64);
        const s = r.m[r.o..][0..n];
        r.o += n;
        return s;
    }
    fn skip(r: *Rd, t: u32) void {
        switch (t) {
            0, 1, 7 => r.o += 1,
            2, 3 => r.o += 2,
            4, 5, 6 => r.o += 4,
            10, 11, 12 => r.o += 8,
            8 => _ = r.str(),
            9 => {
                const et = r.int(u32);
                const n = r.int(u64);
                for (0..n) |_| r.skip(et);
            },
            else => unreachable,
        }
    }
};
