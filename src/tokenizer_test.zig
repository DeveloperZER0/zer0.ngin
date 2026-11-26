// zig run -O ReleaseFast -mcpu=znver5 src/tokenizer_test.zig
const std = @import("std");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Alloc = std.mem.Allocator;
const Io = std.Io;
// model: INKLING_MODEL, else the GGUF in %LOCALAPPDATA%\Programs\llama.cpp-Inkling\models
var model: []const u8 = "";
const oracle = "C:/src/ref/build/bin/llama-tokenize.exe";

const Header = struct {
    rest: []const u8,
    fn take(h: *Header, n: usize) ![]const u8 {
        if (h.rest.len < n) return error.ShortHeader;
        const data = h.rest[0..n];
        h.rest = h.rest[n..];
        return data;
    }
    fn int(h: *Header, comptime T: type) !T {
        return std.mem.readInt(T, (try h.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }
    fn string(h: *Header) ![]const u8 {
        return h.take(try h.int(u64));
    }
    fn skip(h: *Header, kind: u32) anyerror!void {
        const size: usize = switch (kind) {
            0, 1, 7 => 1,
            2, 3 => 2,
            4, 5, 6 => 4,
            10, 11, 12 => 8,
            8 => {
                _ = try h.string();
                return;
            },
            9 => {
                const subtype = try h.int(u32);
                const count = try h.int(u64);
                for (0..count) |_| try h.skip(subtype);
                return;
            },
            else => return error.UnknownMetadataType,
        };
        _ = try h.take(size);
    }
    fn arrayLen(h: *Header, kind: u32, expected: u32) !usize {
        if (kind != 9 or try h.int(u32) != expected) return error.WrongMetadataType;
        return h.int(u64);
    }
    fn strings(h: *Header, a: Alloc, kind: u32) ![][]const u8 {
        const items = try a.alloc([]const u8, try h.arrayLen(kind, 8));
        for (items) |*s| s.* = try h.string();
        return items;
    }
};

const Vocab = struct { tokens: [][]const u8, types: []i32, merges: [][]const u8 };
fn load(a: Alloc, io: Io) !Vocab {
    const f = try Io.Dir.openFileAbsolute(io, model, .{});
    defer f.close(io);
    const buf = try a.alloc(u8, 12 * 1024 * 1024);
    var h = Header{ .rest = buf[0..try f.readPositionalAll(io, buf, 0)] };
    if (!std.mem.eql(u8, try h.take(4), "GGUF") or try h.int(u32) != 3) return error.BadHeader;
    _ = try h.int(u64);
    const count = try h.int(u64);
    var toks: ?[][]const u8 = null;
    var types: ?[]i32 = null;
    var merges: ?[][]const u8 = null;
    for (0..count) |_| {
        const key = try h.string();
        const kind = try h.int(u32);
        if (std.mem.eql(u8, key, "tokenizer.ggml.tokens")) {
            toks = try h.strings(a, kind);
        } else if (std.mem.eql(u8, key, "tokenizer.ggml.token_type")) {
            types = try a.alloc(i32, try h.arrayLen(kind, 5));
            for (types.?) |*t| t.* = try h.int(i32);
        } else if (std.mem.eql(u8, key, "tokenizer.ggml.merges")) {
            merges = try h.strings(a, kind);
        } else try h.skip(kind);
        if (toks != null and types != null and merges != null) return .{ .tokens = toks.?, .types = types.?, .merges = merges.? };
    }
    return error.MissingVocabulary;
}

const Suite = struct {
    a: Alloc,
    io: Io,
    env: *std.process.Environ.Map,
    tok: *const Tokenizer,
    tmp: []const u8,
    passed: [2]usize = .{ 0, 0 },
    failed: [2]usize = .{ 0, 0 },

    fn check(t: *Suite, input: []const u8, special: bool) !void {
        const invalid: usize = @intFromBool(!std.unicode.utf8ValidateSlice(input));
        const start = Io.Clock.awake.now(t.io);
        const ids = try t.tok.encode(t.a, input, special);
        defer t.a.free(ids);
        const elapsed = start.durationTo(Io.Clock.awake.now(t.io)).toNanoseconds();
        if (input.len >= 100000) std.debug.print("Document {d} bytes, {d} IDs, {d:.2} MB/s\n", .{ input.len, ids.len, @as(f64, @floatFromInt(input.len * 1000)) / @as(f64, @floatFromInt(elapsed)) });
        var ok = true;
        if (invalid == 0) {
            var pos: usize = 0;
            for (ids) |id| {
                const p = t.tok.piece(id);
                if (!std.mem.startsWith(u8, input[pos..], p)) {
                    ok = false;
                    break;
                }
                pos += p.len;
            }
            ok = ok and pos == input.len;
        }
        try Io.Dir.cwd().writeFile(t.io, .{ .sub_path = t.tmp, .data = input });
        const argv = [_][]const u8{ oracle, "-m", model, "-f", t.tmp, "--ids", "--no-escape", "--no-bos", "--no-parse-special" };
        const result = try std.process.run(t.a, t.io, .{ .argv = argv[0 .. argv.len - @intFromBool(special)], .environ_map = t.env });
        defer t.a.free(result.stdout);
        defer t.a.free(result.stderr);
        if (result.term != .exited or result.term.exited != 0) {
            std.debug.print("Oracle failed {any}: {s}\n", .{ result.term, result.stderr });
            return error.OracleFailed;
        }
        const trimmed = std.mem.trimEnd(u8, result.stdout, " \t\r\n");
        const last = if (std.mem.lastIndexOfScalar(u8, trimmed, '\n')) |n| trimmed[n + 1 ..] else trimmed;
        if (last.len < 2 or last[0] != '[' or last[last.len - 1] != ']') return error.BadOracleOutput;
        var iter = std.mem.tokenizeAny(u8, last[1 .. last.len - 1], " ,");
        var i: usize = 0;
        while (iter.next()) |num| : (i += 1) {
            const id = try std.fmt.parseInt(u32, num, 10);
            if (i >= ids.len or id != ids[i]) {
                ok = false;
                break;
            }
        }
        ok = ok and i == ids.len;
        if (ok) t.passed[invalid] += 1 else {
            t.failed[invalid] += 1;
            std.debug.print("Mismatch at ID {d}, input prefix {x}, special={any}\n", .{ i, input[0..@min(input.len, 60)], special });
        }
    }
};

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const tmp_alloc = arena.allocator();
    model = init.environ_map.get("INKLING_MODEL") orelse try std.fmt.allocPrint(tmp_alloc, "{s}/Programs/llama.cpp-Inkling/models/Inkling-2-Mini-UD-Q4_K_M.gguf", .{init.environ_map.get("LOCALAPPDATA") orelse "."});
    const v = try load(tmp_alloc, io);
    const before = Io.Clock.awake.now(io);
    var tok = try Tokenizer.init(a, v.tokens, v.types, v.merges);
    defer tok.deinit();
    std.debug.print("Init {d} ms, {d} tokens, {d} merges\n", .{ before.durationTo(Io.Clock.awake.now(io)).toMilliseconds(), v.tokens.len, v.merges.len });
    for (v.tokens, v.types, 0..) |s, typ, i| {
        if (tok.isSpecial(@intCast(i)) != (typ == 3 or typ == 4)) return error.SpecialTypeMismatch;
        if (tok.isSpecial(@intCast(i)) and (tok.find(s) != @as(u32, @intCast(i)) or !std.mem.eql(u8, tok.piece(@intCast(i)), s))) return error.SpecialLookupMismatch;
    }
    if (tok.find("hello") != null) return error.SpecialLookupMismatch;
    const path = try std.fmt.allocPrint(tmp_alloc, "C:\\msys64\\mingw64\\bin;{s}", .{init.environ_map.get("PATH") orelse ""});
    try init.environ_map.put("PATH", path);
    var nonce: [12]u8 = undefined;
    io.random(&nonce);
    const tmp = try std.fs.path.join(tmp_alloc, &.{ init.environ_map.get("TEMP") orelse ".", try std.fmt.allocPrint(tmp_alloc, "inkling-oracle-{x}.txt", .{&nonce}) });
    const file = try Io.Dir.cwd().createFile(io, tmp, .{ .exclusive = true });
    file.close(io);
    defer Io.Dir.cwd().deleteFile(io, tmp) catch {};
    var suite = Suite{ .a = a, .io = io, .env = init.environ_map, .tok = &tok, .tmp = tmp };
    const samples = [_][]const u8{
        "",                                      " ",                          "\n",                                         "\r\n",                                                   "\t",                             "hello",             " hello",          "Hello, world!", "a  b   c",    "a   ",                 "   \t\n \n   a",   "\n\t\r  x",                      "\t\t  ",                                             "a\r\n\r\nb",
        "Zażółć gęślą jaźń.",
        "ŁÓDŹ łódź ŁoDź",
        "ŻABCáXYZ",
        "camelCaseHTTPServer",
        "AéBCäDE",
        "ABCdEFGH",
        "ABC東京DE",
        "can't WON'T I'LL we've you'd",          "he's it's 'LL 'RE",
        "I’M they’ll she’d",
        "A'S B'T C'RE D'VE E'M F'LL G'D",        "split'rx 'l 'v",             "1234567890123456789",
        "x１２３４５６y",
        "٠١٢٣٤٥٦۷۸۹",
        "Ⅷ¼²³42",
        "0.0123e-456789",                        "a\u{301}b",                  "\u{301}\u{308}Z",
        "कर्म नमस्ते",
        "ภาษาไทย",
        "עברית",
        "العَرَبِيَّة",
        "Съешь ещё этих мягких французских булок.",
        "中文，標點。日本語かな",
        "한글 테스트",
        "©️🚀👩🏽‍💻👨‍👩‍👧‍👦",
        "🇵🇱 🏳️‍🌈",
        "1️⃣2️⃣",
        "𐐀𐐨𠀀",
        "a\u{85}b\u{a0}c\u{2003}d\u{2028}e",     "\u{3000}\u{3000}a",          "if x:\r\n    print('hi')\r\n\treturn 0\r\n", "#include <stdio.h>\nint main(void) {\n\treturn 0;\n}\n", "{\"a\":[1,12,1234],\"b\":null}", "\\n\\t\\\\\\u0020", "</p>//\r\n/word", "  ::/./\n//a",  "\x00abc\x00", "\x01\x0b\x0c\x1f\x7f", "<|message_user|>", " before<|message_model|>after ", "<|message_user|><|content_text|>Hi!<|end_message|>", "<|unused_200055|><|unused|>",
        "<|content_invoke_tool_json|>{\"x\":1}", "<|message_use|><|unknown|>", "<<|message_user|>>",                         "__________",                                             " a---b...c ",
        "e\u{301} é E\u{301}",
        "πABCπDEF",
        "\u{301}ABCd",                           "\u{301}ABC",                 "AB\u{301}CD",
        "😀's",
    };
    for (samples) |s| try suite.check(s, true);
    for (samples) |s| if (std.mem.indexOf(u8, s, "<|") != null) try suite.check(s, false);
    var random = std.Random.DefaultPrng.init(0xacf931);
    const seed = "Zażółć! 👩‍💻 中文 Русский a\u{301} בּ";
    for (0..16) |_| {
        var buf = seed.*;
        buf[random.random().uintLessThan(usize, buf.len)] ^= random.random().intRangeAtMost(u8, 1, 255);
        try suite.check(&buf, true);
    }
    try suite.check("\xed\xa0\x80\xed\xb0\x80", true);
    try suite.check("\xf0\x80\x80\x80\xe2\x82\xc0\xaf\xff", false);
    try suite.check("A" ** 512 ++ "éBC" ++ "z" ** 512, true);

    // A multilingual transcript and a CRLF code fixture with varied identifiers
    // and numeric fields. Keep each document on a complete line boundary.
    for (0..2) |doc_index| {
        var doc: std.ArrayList(u8) = .empty;
        defer doc.deinit(a);
        while (doc.items.len < 102400) {
            const sample = samples[random.random().uintLessThan(usize, samples.len)];
            if (doc_index == 0) {
                try doc.appendSlice(a, "<|message_user|><|content_text|>");
                try doc.appendSlice(a, sample);
                try doc.appendSlice(a, "<|end_message|>\n");
            } else {
                const names = [_][]const u8{ "request_id", "line", "position", "n", "packetCount" };
                const name = names[random.random().uintLessThan(usize, names.len)];
                const value = random.random().int(u32);
                try doc.print(a, "    if ({s} >= {d}) {{\r\n        write({s});\r\n    }}\r\n", .{ name, value, name });
            }
        }
        try suite.check(doc.items, true);
    }
    std.debug.print("Valid UTF-8: {d} passed, {d} failed. Invalid UTF-8: {d} passed, {d} failed.\n", .{ suite.passed[0], suite.failed[0], suite.passed[1], suite.failed[1] });
    if (suite.failed[0] + suite.failed[1] != 0) return error.OracleMismatch;
}
