const std = @import("std");
const unicode = @import("unicode_tables.zig");
const Alloc = std.mem.Allocator;
const none = std.math.maxInt(u32);

const reverse_alphabet = blk: {
    var table = [_]u8{0} ** 324;
    var extra: usize = 256;
    for (0..256) |b| {
        if ((b >= 33 and b <= 126) or (b >= 161 and b <= 172) or b >= 174) {
            table[b] = @intCast(b);
        } else {
            table[extra] = @intCast(b);
            extra += 1;
        }
    }
    break :blk table;
};

// GPT-2 maps bytes onto ASCII graphic characters and two-byte UTF-8 scalars.
fn decodeAlphabet(dst: []u8, src: []const u8) ![]u8 {
    var i: usize = 0;
    var n: usize = 0;
    while (i < src.len) : (n += 1) {
        const a = src[i];
        i += 1;
        if (a >= 33 and a <= 126) {
            dst[n] = a;
        } else {
            if (a < 0xc2 or a > 0xc5 or i == src.len or src[i] & 0xc0 != 0x80) return error.InvalidAlphabet;
            const cp = (@as(u16, a & 31) << 6) | (src[i] & 63);
            if (cp < 161 or cp == 173 or cp >= reverse_alphabet.len) return error.InvalidAlphabet;
            dst[n] = reverse_alphabet[cp];
            i += 1;
        }
    }
    return dst[0..n];
}

const Rule = struct { rank: u32, id: u32 };
fn pair(l: u32, r: u32) u64 {
    return (@as(u64, l) << 32) | r;
}
pub const Tokenizer = struct {
    allocator: Alloc,
    types: []const i32,
    pieces: [][]const u8,
    storage: []u8,
    ids: std.StringHashMapUnmanaged(u32) = .empty,
    rules: std.AutoHashMapUnmanaged(u64, Rule) = .empty,
    specials: std.ArrayList(u32) = .empty,
    starts: [256]bool = @splat(false),
    bytes: [256]u32 = @splat(none),

    pub fn init(alloc: Alloc, tokens: []const []const u8, token_types: []const i32, merges: []const []const u8) !Tokenizer {
        if (tokens.len != token_types.len or tokens.len >= none or merges.len >= none) return error.InvalidVocabulary;
        var size: usize = 0;
        for (tokens) |s| size = try std.math.add(usize, size, s.len);
        const storage = try alloc.alloc(u8, size);
        const pieces = alloc.alloc([]const u8, tokens.len) catch |err| {
            alloc.free(storage);
            return err;
        };
        var self = Tokenizer{ .allocator = alloc, .types = token_types, .storage = storage, .pieces = pieces };
        errdefer self.deinit();
        try self.ids.ensureTotalCapacity(alloc, @intCast(tokens.len));
        var used: usize = 0;
        for (tokens, token_types, 0..) |s, t, idx| {
            const id: u32 = @intCast(idx);
            if (s.len == 0) return error.EmptyToken;
            if (t == 2 or t == 3 or t == 4) {
                pieces[idx] = s;
                self.starts[s[0]] = true;
                try self.specials.append(alloc, id);
                // The whole-word shortcut looks up byte-encoded text, including controls.
                const key = decodeAlphabet(storage[used..], s) catch continue;
                self.ids.putAssumeCapacity(key, id);
                used += key.len;
            } else {
                const key = try decodeAlphabet(storage[used..], s);
                used += key.len;
                pieces[idx] = key;
                self.ids.putAssumeCapacity(key, id);
                if (key.len == 1) self.bytes[key[0]] = id;
            }
        }
        std.mem.sort(u32, self.specials.items, pieces, longer);
        try self.rules.ensureTotalCapacity(alloc, @intCast(merges.len));
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(alloc);
        for (merges, 0..) |s, rank| {
            const sep = std.mem.indexOfScalar(u8, s, ' ') orelse return error.InvalidMerge;
            try buf.resize(alloc, s.len);
            const left = try decodeAlphabet(buf.items, s[0..sep]);
            const right = try decodeAlphabet(buf.items[left.len..], s[sep + 1 ..]);
            const l = self.ids.get(left) orelse return error.MissingMergeToken;
            const r = self.ids.get(right) orelse return error.MissingMergeToken;
            const merged = self.ids.get(buf.items[0 .. left.len + right.len]) orelse return error.MissingMergeToken;
            const entry = self.rules.getOrPutAssumeCapacity(pair(l, r));
            if (!entry.found_existing) entry.value_ptr.* = .{ .rank = @intCast(rank), .id = merged };
        }
        return self;
    }

    fn longer(pieces: [][]const u8, a: u32, b: u32) bool {
        return pieces[a].len > pieces[b].len;
    }

    pub fn deinit(self: *Tokenizer) void {
        self.rules.deinit(self.allocator);
        self.ids.deinit(self.allocator);
        self.specials.deinit(self.allocator);
        self.allocator.free(self.pieces);
        self.allocator.free(self.storage);
        self.* = undefined;
    }

    pub fn piece(self: *const Tokenizer, id: u32) []const u8 {
        return if (id < self.pieces.len) self.pieces[id] else "";
    }

    pub fn isSpecial(self: *const Tokenizer, id: u32) bool {
        return id < self.types.len and (self.types[id] == 3 or self.types[id] == 4);
    }

    pub fn find(self: *const Tokenizer, text: []const u8) ?u32 {
        for (self.specials.items) |id| {
            if (self.isSpecial(id) and std.mem.eql(u8, text, self.piece(id))) return id;
        }
        return null;
    }

    pub fn encode(self: *const Tokenizer, alloc: Alloc, text: []const u8, parse_special: bool) ![]u32 {
        var work = Work{ .alloc = alloc, .tok = self };
        defer work.nodes.deinit(alloc);
        defer work.queue.deinit(alloc);
        defer work.out.deinit(alloc);
        var start: usize = 0;
        var pos: usize = 0;
        while (pos < text.len) {
            if (self.starts[text[pos]]) {
                for (self.specials.items) |id| {
                    if (!parse_special and self.types[id] != 4) continue;
                    const literal = self.piece(id);
                    if (std.mem.startsWith(u8, text[pos..], literal)) {
                        try work.fragment(text[start..pos]);
                        try work.out.append(alloc, id);
                        pos += literal.len;
                        start = pos;
                        break;
                    }
                } else {
                    pos += 1;
                }
            } else pos += 1;
        }
        try work.fragment(text[start..]);
        return work.out.toOwnedSlice(alloc);
    }
};

const Category = enum { other, number, upper, lower, letter, mark, space, newline, end };
const Rune = struct { cat: Category, len: usize };
fn rune(s: []const u8, i: usize) Rune {
    if (i == s.len) return .{ .cat = .end, .len = 0 };
    const b = s[i];
    if (b < 128) return .{ .len = 1, .cat = switch (b) {
        'A'...'Z' => .upper,
        'a'...'z' => .lower,
        '0'...'9' => .number,
        '\r', '\n' => .newline,
        ' ', '\t', 11, 12 => .space,
        else => .other,
    } };
    const len = std.unicode.utf8ByteSequenceLength(b) catch unreachable;
    const cp = std.unicode.wtf8Decode(s[i..][0..len]) catch unreachable;
    return .{ .len = len, .cat = switch (unicode.category(cp)) {
        1 => .number,
        2 => .letter,
        3 => .mark,
        4 => .space,
        else => .other,
    } };
}

fn low(c: Category) bool {
    return c == .lower or c == .letter or c == .mark;
}

const Word = struct { first: usize, second: usize };
// llama.cpp collapses non-ASCII L and M before applying its [^a-z] and
// [^A-Z] lookaheads. Both classes therefore include all non-ASCII letters.
fn word(s: []const u8, start: usize) Word {
    var i = start;
    var shared_end: usize = 0;
    var r = rune(s, i);
    while (r.cat == .upper or r.cat == .letter or r.cat == .mark) {
        i += r.len;
        if (low(r.cat)) shared_end = i;
        r = rune(s, i);
    }
    const upper_end = i;
    while (low(r.cat)) {
        i += r.len;
        r = rune(s, i);
    }
    return .{ .first = if (i > upper_end) i else shared_end, .second = if (upper_end > start) i else 0 };
}

fn split(s: []const u8) usize {
    const first = rune(s, 0);
    const prefix = first.cat == .space or first.cat == .other or first.cat == .mark;
    const pref = word(s, if (prefix) first.len else 0);
    var end = pref.first;
    if (end == 0) {
        // Only a mark can be both the optional prefix and part of the word.
        const bare = if (first.cat == .mark) word(s, 0) else Word{ .first = 0, .second = 0 };
        end = if (bare.first != 0) bare.first else if (pref.second != 0) pref.second else bare.second;
    }
    if (end != 0) {
        if (s.len - end >= 2 and s[end] == '\'') {
            switch (std.ascii.toLower(s[end + 1])) {
                's', 't', 'm', 'd' => return end + 2,
                'r', 'v', 'l' => |c| {
                    if (s.len - end >= 3 and std.ascii.toLower(s[end + 2]) == (if (c == 'l') @as(u8, 'l') else 'e')) return end + 3;
                },
                else => {},
            }
        }
        return end;
    }
    if (first.cat == .number) {
        end = first.len;
        for (0..2) |_| {
            const r = rune(s, end);
            if (r.cat != .number) break;
            end += r.len;
        }
        return end;
    }
    var i: usize = @intFromBool(s[0] == ' ');
    var r = rune(s, i);
    if (r.cat == .other or r.cat == .mark) {
        while (r.cat == .other or r.cat == .mark) {
            i += r.len;
            r = rune(s, i);
        }
        while (i < s.len and (s[i] == '\r' or s[i] == '\n' or s[i] == '/')) i += 1;
        return i;
    }
    i = 0;
    r = first;
    var last_newline: usize = 0;
    var last_start: usize = 0;
    while (r.cat == .space or r.cat == .newline) {
        last_start = i;
        i += r.len;
        if (r.cat == .newline) last_newline = i;
        r = rune(s, i);
    }
    if (last_newline != 0) return last_newline;
    return if (i == s.len or last_start == 0) i else last_start;
}

const Node = struct { id: u32, prev: u32, next: u32 };
const Candidate = struct {
    left: u32,
    right: u32,
    rule: Rule,
    fn order(_: void, a: Candidate, b: Candidate) std.math.Order {
        return if (a.rule.rank == b.rule.rank) std.math.order(a.left, b.left) else std.math.order(a.rule.rank, b.rule.rank);
    }
};

const Work = struct {
    alloc: Alloc,
    tok: *const Tokenizer,
    nodes: std.ArrayList(Node) = .empty,
    queue: std.PriorityQueue(Candidate, void, Candidate.order) = .empty,
    out: std.ArrayList(u32) = .empty,

    fn fragment(w: *Work, input: []const u8) !void {
        var cleaned: std.ArrayList(u8) = .empty;
        defer cleaned.deinit(w.alloc);
        var s = input;
        // llama.cpp accepts encoded surrogate code points and replaces each other invalid byte individually.
        if (!std.unicode.utf8ValidateSlice(s)) {
            var i: usize = 0;
            while (i < input.len) {
                const n = std.unicode.utf8ByteSequenceLength(input[i]) catch 0;
                if (n != 0 and n <= input.len - i and std.unicode.wtf8ValidateSlice(input[i..][0..n])) {
                    try cleaned.appendSlice(w.alloc, input[i..][0..n]);
                    i += n;
                } else {
                    try cleaned.appendSlice(w.alloc, "\u{fffd}");
                    i += 1;
                }
            }
            s = cleaned.items;
        }
        while (s.len != 0) {
            const n = split(s);
            try w.bpe(s[0..n]);
            s = s[n..];
        }
    }

    fn offer(w: *Work, l: u32, r: u32) !void {
        if (l == none or r == none) return;
        const rule = w.tok.rules.get(pair(w.nodes.items[l].id, w.nodes.items[r].id)) orelse return;
        try w.queue.push(w.alloc, .{ .left = l, .right = r, .rule = rule });
    }

    fn bpe(w: *Work, s: []const u8) !void {
        if (w.tok.ids.get(s)) |id| return w.out.append(w.alloc, id);
        if (s.len >= none) return error.WordTooLong;
        try w.nodes.resize(w.alloc, s.len);
        for (w.nodes.items, s, 0..) |*node, b, i| {
            const idx: u32 = @intCast(i);
            const id = w.tok.bytes[b];
            if (id == none) return error.MissingByteToken;
            node.* = .{ .id = id, .prev = idx -% 1, .next = if (i + 1 == s.len) none else idx + 1 };
            try w.offer(node.prev, idx);
        }
        while (w.queue.pop()) |c| {
            const l = &w.nodes.items[c.left];
            const r = &w.nodes.items[c.right];
            if (l.next != c.right or r.id == none) continue;
            // Merges strictly increase the byte length of a symbol.
            if (w.tok.piece(l.id).len + w.tok.piece(r.id).len != w.tok.piece(c.rule.id).len) continue;
            l.id = c.rule.id;
            l.next = r.next;
            r.id = none;
            r.next = none;
            if (l.next != none) w.nodes.items[l.next].prev = c.left;
            try w.offer(l.prev, c.left);
            try w.offer(c.left, l.next);
        }
        var idx: u32 = 0;
        while (idx != none) {
            const node = w.nodes.items[idx];
            try w.out.append(w.alloc, node.id);
            idx = node.next;
        }
    }
};
