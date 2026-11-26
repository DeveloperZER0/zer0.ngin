const std = @import("std");
const server = @import("server.zig");
const chat_template = @import("chat_template.zig");

extern "kernel32" fn Sleep(dwMilliseconds: u32) callconv(.winapi) void;

const SpecialEntry = struct {
    literal: []const u8,
    id: u32,
};

const specials = [_]SpecialEntry{
    .{ .literal = chat_template.TOK_MSG_USER, .id = 1000 },
    .{ .literal = chat_template.TOK_MSG_MODEL, .id = 1001 },
    .{ .literal = chat_template.TOK_MSG_SYSTEM, .id = 1002 },
    .{ .literal = chat_template.TOK_MSG_TOOL, .id = 1003 },
    .{ .literal = chat_template.TOK_CONTENT_THINKING, .id = 1004 },
    .{ .literal = chat_template.TOK_CONTENT_TEXT, .id = 1005 },
    .{ .literal = chat_template.TOK_CONTENT_XML, .id = 1006 },
    .{ .literal = chat_template.TOK_CONTENT_IMAGE, .id = 1007 },
    .{ .literal = chat_template.TOK_CONTENT_TOOL_JSON, .id = 1008 },
    .{ .literal = chat_template.TOK_END_MESSAGE, .id = 1009 },
    .{ .literal = chat_template.TOK_END_SAMPLING, .id = 1010 },
};

const TOK_RUN_A64: u32 = 256;

const FakeEngine = struct {
    active_gens: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    max_concurrent: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    aborted_count: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    completed_count: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    fn special(_: *anyopaque, literal: []const u8) ?u32 {
        for (specials) |s| {
            if (std.mem.eql(u8, literal, s.literal)) return s.id;
        }
        return null;
    }

    fn isEog(_: *anyopaque, id: u32) bool {
        return id == 1010;
    }

    fn tokenize(_: *anyopaque, alloc: std.mem.Allocator, text: []const u8, parse_special: bool) anyerror![]u32 {
        var toks: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        while (i < text.len) {
            if (parse_special and text[i] == '<') {
                var matched: ?SpecialEntry = null;
                for (specials) |s| {
                    if (std.mem.startsWith(u8, text[i..], s.literal)) {
                        matched = s;
                        break;
                    }
                }
                if (matched) |m| {
                    try toks.append(alloc, m.id);
                    i += m.literal.len;
                    continue;
                }
            }

            // Compress runs of 64 'A's into a single token so long-body tests (e.g. 256 KB) stay under n_ctx,
            // while huge prompts (> 512 KB of 'A's) deterministically exceed n_ctx.
            if (text[i] == 'A' and i + 64 <= text.len) {
                var all_a = true;
                for (text[i .. i + 64]) |c| {
                    if (c != 'A') {
                        all_a = false;
                        break;
                    }
                }
                if (all_a) {
                    try toks.append(alloc, TOK_RUN_A64);
                    i += 64;
                    continue;
                }
            }

            try toks.append(alloc, text[i]);
            i += 1;
        }
        return toks.toOwnedSlice(alloc);
    }

    fn decodePrompt(buf: []u8, prompt: []const u32) []const u8 {
        var len: usize = 0;
        for (prompt) |t| {
            if (t < 256) {
                if (len < buf.len) {
                    buf[len] = @intCast(t);
                    len += 1;
                }
            } else if (t == TOK_RUN_A64) {
                if (len < buf.len) {
                    buf[len] = 'A';
                    len += 1;
                }
            } else {
                const tag: []const u8 = switch (t) {
                    1000 => "[U]",
                    1001 => "[M]",
                    1002 => "[S]",
                    1003 => "[T]",
                    1004 => "[TH]",
                    1005 => "[TX]",
                    1006 => "[XML]",
                    1007 => "[IMG]",
                    1008 => "[TOOL]",
                    1009 => "[END]",
                    1010 => "[EOS]",
                    else => "[?]",
                };
                if (len + tag.len <= buf.len) {
                    @memcpy(buf[len .. len + tag.len], tag);
                    len += tag.len;
                }
            }
        }
        return buf[0..len];
    }

    const TokEmit = struct {
        id: u32,
        bytes: []const u8,
    };

    fn emitSeq(
        seq: []const TokEmit,
        max_tokens: u32,
        delay_ms: u32,
        user: *anyopaque,
        on_token: *const fn (user: *anyopaque, id: u32, bytes: []const u8) bool,
        emitted_out: *u32,
    ) server.FinishReason {
        var count: u32 = 0;
        for (seq) |item| {
            if (count >= max_tokens) {
                emitted_out.* = count;
                return .length;
            }
            if (delay_ms > 0) Sleep(delay_ms);
            count += 1;
            const keep_going = on_token(user, item.id, item.bytes);
            if (item.id == 1010) {
                emitted_out.* = count;
                return .stop;
            }
            if (!keep_going) {
                emitted_out.* = count;
                return .aborted;
            }
        }
        emitted_out.* = count;
        return .stop;
    }

    fn generate(
        ctx_ptr: *anyopaque,
        prompt: []const u32,
        params: server.GenParams,
        stats: *server.GenStats,
        user: *anyopaque,
        on_token: *const fn (user: *anyopaque, id: u32, bytes: []const u8) bool,
    ) anyerror!server.FinishReason {
        const self: *FakeEngine = @ptrCast(@alignCast(ctx_ptr));

        const cur = self.active_gens.fetchAdd(1, .seq_cst) + 1;
        // Track peak concurrency to verify single-slot mutex
        var prev_max = self.max_concurrent.load(.seq_cst);
        while (cur > prev_max) {
            if (self.max_concurrent.cmpxchgWeak(prev_max, cur, .seq_cst, .seq_cst)) |actual| {
                prev_max = actual;
            } else break;
        }
        defer _ = self.active_gens.fetchSub(1, .seq_cst);

        var decoded_buf: [8192]u8 = undefined;
        const decoded = decodePrompt(&decoded_buf, prompt);

        stats.prompt_tokens = @intCast(prompt.len);
        stats.cached_tokens = if (params.cache_prompt and prompt.len >= 4) 4 else 0;
        stats.prompt_ms = 12.5;
        stats.decode_ms = 25.0;

        var emitted: u32 = 0;
        var reason: server.FinishReason = .stop;

        const is_chat = prompt.len > 0 and (prompt[0] == 1002 or prompt[0] == 1000);

        if (!is_chat) {
            // Raw /v1/completions path
            const seq = [_]TokEmit{
                .{ .id = 10, .bytes = "Alpha " },
                .{ .id = 11, .bytes = "Beta " },
                .{ .id = 12, .bytes = "ST" },
                .{ .id = 13, .bytes = "OP_H" },
                .{ .id = 14, .bytes = "ERE" },
                .{ .id = 15, .bytes = " Gamma" },
                .{ .id = 1010, .bytes = "" },
            };
            reason = emitSeq(&seq, params.max_tokens, 0, user, on_token, &emitted);
        } else if (std.mem.indexOf(u8, decoded, "CASE_UTF8") != null) {
            // Split multi-byte UTF-8 ("Zażółć 🚀") across individual byte tokens!
            // "ż" = C5 BC, "ó" = C3 B3, "ł" = C5 82, "ć" = C4 87, "🚀" = F0 9F 9A 80
            const seq = [_]TokEmit{
                .{ .id = 1001, .bytes = chat_template.TOK_MSG_MODEL },
                .{ .id = 1004, .bytes = chat_template.TOK_CONTENT_THINKING },
                .{ .id = 20, .bytes = "My\xc5" },
                .{ .id = 21, .bytes = "\x9bl\xc4" },
                .{ .id = 22, .bytes = "\x99" },
                .{ .id = 1009, .bytes = chat_template.TOK_END_MESSAGE },
                .{ .id = 1001, .bytes = chat_template.TOK_MSG_MODEL },
                .{ .id = 1005, .bytes = chat_template.TOK_CONTENT_TEXT },
                .{ .id = 23, .bytes = "Za\xc5" },
                .{ .id = 24, .bytes = "\xbc\xc3" },
                .{ .id = 25, .bytes = "\xb3\xc5" },
                .{ .id = 26, .bytes = "\x82\xc4" },
                .{ .id = 27, .bytes = "\x87 " },
                .{ .id = 28, .bytes = "\xf0" },
                .{ .id = 29, .bytes = "\x9f" },
                .{ .id = 30, .bytes = "\x9a" },
                .{ .id = 31, .bytes = "\x80" },
                .{ .id = 1009, .bytes = chat_template.TOK_END_MESSAGE },
                .{ .id = 1010, .bytes = chat_template.TOK_END_SAMPLING },
            };
            reason = emitSeq(&seq, params.max_tokens, 0, user, on_token, &emitted);
        } else if (std.mem.indexOf(u8, decoded, "CASE_STOP") != null) {
            const seq = [_]TokEmit{
                .{ .id = 1001, .bytes = chat_template.TOK_MSG_MODEL },
                .{ .id = 1005, .bytes = chat_template.TOK_CONTENT_TEXT },
                .{ .id = 40, .bytes = "Hello " },
                .{ .id = 41, .bytes = "ST" },
                .{ .id = 42, .bytes = "OP_H" },
                .{ .id = 43, .bytes = "ERE" },
                .{ .id = 44, .bytes = " ignored tail" },
                .{ .id = 1009, .bytes = chat_template.TOK_END_MESSAGE },
                .{ .id = 1010, .bytes = chat_template.TOK_END_SAMPLING },
            };
            reason = emitSeq(&seq, params.max_tokens, 0, user, on_token, &emitted);
        } else if (std.mem.indexOf(u8, decoded, "CASE_TOOL") != null) {
            const seq = [_]TokEmit{
                .{ .id = 1001, .bytes = chat_template.TOK_MSG_MODEL },
                .{ .id = 1004, .bytes = chat_template.TOK_CONTENT_THINKING },
                .{ .id = 50, .bytes = "Need " },
                .{ .id = 51, .bytes = "weather." },
                .{ .id = 1009, .bytes = chat_template.TOK_END_MESSAGE },
                .{ .id = 1001, .bytes = chat_template.TOK_MSG_MODEL },
                .{ .id = 52, .bytes = "get_weather" },
                .{ .id = 1008, .bytes = chat_template.TOK_CONTENT_TOOL_JSON },
                .{ .id = 53, .bytes = "{\"name\":\"get_weather\"," },
                .{ .id = 54, .bytes = "\"args\":{\"city\":\"Warsaw\"}}" },
                .{ .id = 1009, .bytes = chat_template.TOK_END_MESSAGE },
                .{ .id = 1010, .bytes = chat_template.TOK_END_SAMPLING },
            };
            reason = emitSeq(&seq, params.max_tokens, 0, user, on_token, &emitted);
        } else if (std.mem.indexOf(u8, decoded, "CASE_SLOW") != null) {
            var seq_buf: [44]TokEmit = undefined;
            seq_buf[0] = .{ .id = 1001, .bytes = chat_template.TOK_MSG_MODEL };
            seq_buf[1] = .{ .id = 1005, .bytes = chat_template.TOK_CONTENT_TEXT };
            for (0..40) |k| {
                seq_buf[2 + k] = .{ .id = @intCast(100 + k), .bytes = "tok " };
            }
            seq_buf[42] = .{ .id = 1009, .bytes = chat_template.TOK_END_MESSAGE };
            seq_buf[43] = .{ .id = 1010, .bytes = chat_template.TOK_END_SAMPLING };
            reason = emitSeq(&seq_buf, params.max_tokens, 25, user, on_token, &emitted);
        } else if (std.mem.indexOf(u8, decoded, "CASE_STATS") != null) {
            var stat_buf: [128]u8 = undefined;
            const s = std.fmt.bufPrint(&stat_buf, "max_concurrent={d};aborted={d};completed={d}", .{
                self.max_concurrent.load(.seq_cst),
                self.aborted_count.load(.seq_cst),
                self.completed_count.load(.seq_cst),
            }) catch "err";
            const seq = [_]TokEmit{
                .{ .id = 1001, .bytes = chat_template.TOK_MSG_MODEL },
                .{ .id = 1005, .bytes = chat_template.TOK_CONTENT_TEXT },
                .{ .id = 60, .bytes = s },
                .{ .id = 1009, .bytes = chat_template.TOK_END_MESSAGE },
                .{ .id = 1010, .bytes = chat_template.TOK_END_SAMPLING },
            };
            reason = emitSeq(&seq, params.max_tokens, 0, user, on_token, &emitted);
        } else if (std.mem.indexOf(u8, decoded, "CASE_ECHO_PROMPT") != null) {
            const seq = [_]TokEmit{
                .{ .id = 1001, .bytes = chat_template.TOK_MSG_MODEL },
                .{ .id = 1005, .bytes = chat_template.TOK_CONTENT_TEXT },
                .{ .id = 61, .bytes = decoded },
                .{ .id = 1009, .bytes = chat_template.TOK_END_MESSAGE },
                .{ .id = 1010, .bytes = chat_template.TOK_END_SAMPLING },
            };
            reason = emitSeq(&seq, params.max_tokens, 0, user, on_token, &emitted);
        } else {
            const seq = [_]TokEmit{
                .{ .id = 1001, .bytes = chat_template.TOK_MSG_MODEL },
                .{ .id = 1004, .bytes = chat_template.TOK_CONTENT_THINKING },
                .{ .id = 70, .bytes = "Step 1. " },
                .{ .id = 71, .bytes = "Step 2." },
                .{ .id = 1009, .bytes = chat_template.TOK_END_MESSAGE },
                .{ .id = 1001, .bytes = chat_template.TOK_MSG_MODEL },
                .{ .id = 1005, .bytes = chat_template.TOK_CONTENT_TEXT },
                .{ .id = 72, .bytes = "Hello, " },
                .{ .id = 73, .bytes = "world!" },
                .{ .id = 1009, .bytes = chat_template.TOK_END_MESSAGE },
                .{ .id = 1010, .bytes = chat_template.TOK_END_SAMPLING },
            };
            reason = emitSeq(&seq, params.max_tokens, 0, user, on_token, &emitted);
        }

        stats.completion_tokens = emitted;
        if (reason == .aborted) {
            _ = self.aborted_count.fetchAdd(1, .seq_cst);
        } else {
            _ = self.completed_count.fetchAdd(1, .seq_cst);
        }
        return reason;
    }
};

pub fn main() !void {
    const alloc = std.heap.smp_allocator;
    var fake: FakeEngine = .{};
    var engine: server.Engine = .{
        .ctx = @ptrCast(&fake),
        .tokenize = FakeEngine.tokenize,
        .generate = FakeEngine.generate,
        .special = FakeEngine.special,
        .is_eog = FakeEngine.isEog,
        .n_ctx = 8192,
    };
    try server.serve(alloc, &engine, "127.0.0.1", 18099, "inkling-test");
}
