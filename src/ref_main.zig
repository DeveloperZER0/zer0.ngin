// ref_main MODEL IDS.txt GEN.txt LOGITS.bin DUMPDIR : teacher-forced comparison of the Zig CPU forward with llama.cpp dumps
const std = @import("std");
const win = @import("win.zig");
const gguf = @import("gguf.zig");
const q = @import("quant.zig");
const cm = @import("cpu_model.zig");

var dump_dir: []const u8 = "";
var dump_on = false;
var ga: std.mem.Allocator = undefined;
var last_pos: usize = 0;
fn cmp(slot: usize, name: []const u8, d: []const f32) void {
    if (!dump_on) return;
    var b: [512]u8 = undefined;
    const il = if (slot == 99) 0 else cm.ggmlIl(slot);
    const path = (if (slot == 99)
        std.fmt.bufPrintSentinel(&b, "{s}/000_{s}.f32", .{ dump_dir, name }, 0)
    else
        std.fmt.bufPrintSentinel(&b, "{s}/000_{s}-{d}.f32", .{ dump_dir, name, il }, 0)) catch return;
    const f = win.readFile(ga, path) catch return;
    defer ga.free(f);
    const all: []const f32 = @as([*]const f32, @ptrCast(@alignCast(f.ptr + 32)))[0 .. (f.len - 32) / 4];
    if (all.len < d.len) return;
    const ref = all[all.len - d.len ..]; // last token
    var num: f64 = 0;
    var den: f64 = 0;
    for (ref, d) |r, v| {
        num += (r - v) * (r - v);
        den += r * r;
    }
    const e = @sqrt(num / (den + 1e-30));
    if (slot < 3 or slot % 12 == 5 or slot == 6 or slot == 18 or slot == 99 or e > 0.08)
        std.debug.print("slot {d:>2} {s:<10} rel err {d:.5}\n", .{ slot, name, e });
}

pub fn ids(a: std.mem.Allocator, path: [:0]const u8) ![]u32 {
    const t = try win.readFile(a, path);
    var l: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, t, " \r\n\t,");
    while (it.next()) |s| try l.append(a, try std.fmt.parseInt(u32, s, 10));
    return l.items;
}
pub fn main(init: std.process.Init) !void {
    const a = std.heap.page_allocator;
    ga = a;
    var args = try init.minimal.args.iterateAllocator(a);
    _ = args.next();
    const model = args.next().?;
    const ip = args.next().?;
    const gp = args.next().?;
    const lp = args.next().?;
    dump_dir = args.next().?;
    const nmax: usize = if (args.next()) |s| try std.fmt.parseInt(usize, s, 10) else 1000;
    var g = try gguf.Gguf.open(a, try a.dupeZ(u8, model));
    var pool: q.Pool = .{ .nt = 12 };
    try pool.start();
    const pr = try ids(a, try a.dupeZ(u8, ip));
    const gen = try ids(a, try a.dupeZ(u8, gp));
    const lgb = try win.mapFile(try a.dupeZ(u8, lp));
    const rl: [*]const f32 = @ptrCast(@alignCast(lgb.ptr));
    var m = try cm.Model.init(a, &g, &pool, pr.len + gen.len + 8);
    m.dump = cmp;
    const logits = try a.alloc(f32, cm.NVU);
    const t0 = win.now();
    for (pr, 0..) |t, i| {
        dump_on = i == pr.len - 1;
        m.forward(t, if (i == pr.len - 1) logits else null);
    }
    dump_on = false;
    std.debug.print("prompt {d} tokens {d:.2} tok/s\n", .{ pr.len, @as(f64, @floatFromInt(pr.len)) / (win.now() - t0) });
    var top1: usize = 0;
    var kl: f64 = 0;
    var n: usize = 0;
    const t1 = win.now();
    for (gen, 0..) |t, i| {
        if (i >= nmax) break;
        const r = rl[i * cm.NVU ..][0..cm.NVU];
        var bi: usize = 0;
        var ri: usize = 0;
        var mz: f32 = -1e9;
        var rz: f32 = -1e9;
        for (0..cm.NVU) |k| {
            if (logits[k] > mz) {
                mz = logits[k];
                bi = k;
            }
            if (r[k] > rz) {
                rz = r[k];
                ri = k;
            }
        }
        var sm: f64 = 0;
        var sr: f64 = 0;
        for (0..cm.NVU) |k| {
            sm += @exp(@as(f64, logits[k]) - mz);
            sr += @exp(@as(f64, r[k]) - rz);
        }
        var kk: f64 = 0;
        for (0..cm.NVU) |k| {
            const lr = @as(f64, r[k]) - rz - @log(sr);
            const lm = @as(f64, logits[k]) - mz - @log(sm);
            kk += @exp(lr) * (lr - lm);
        }
        kl += kk;
        if (bi == ri) top1 += 1 else std.debug.print("step {d}: ref {d} ({d:.3}) ours {d} ({d:.3}), ours@ref {d:.3} kl {d:.4}\n", .{ i, ri, rz, bi, mz, logits[ri], kk });
        n += 1;
        m.forward(t, logits);
    }
    std.debug.print("teacher-forced top1 {d}/{d}, mean KL {d:.5}, {d:.2} tok/s\n", .{ top1, n, kl / @as(f64, @floatFromInt(n)), @as(f64, @floatFromInt(n)) / (win.now() - t1) });
}
