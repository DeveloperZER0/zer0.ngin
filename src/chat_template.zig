const std = @import("std");
const Allocator = std.mem.Allocator;

pub const TOK_MSG_USER = "<|message_user|>";
pub const TOK_MSG_MODEL = "<|message_model|>";
pub const TOK_MSG_SYSTEM = "<|message_system|>";
pub const TOK_MSG_TOOL = "<|message_tool|>";
pub const TOK_CONTENT_THINKING = "<|content_thinking|>";
pub const TOK_CONTENT_TEXT = "<|content_text|>";
pub const TOK_CONTENT_XML = "<|content_xml|>";
pub const TOK_CONTENT_IMAGE = "<|content_image|>";
pub const TOK_CONTENT_TOOL_JSON = "<|content_invoke_tool_json|>";
pub const TOK_END_MESSAGE = "<|end_message|>";
pub const TOK_END_SAMPLING = "<|content_model_end_sampling|>";

pub const Segment = struct {
    text: []const u8,
    parse_special: bool,
};

pub const RenderError = error{
    OutOfMemory,
    InvalidInput,
    UnknownRole,
    UnknownReasoningEffort,
    ReasoningEffortOutOfRange,
    ImagesNotSupported,
    AudioNotSupported,
    UnsupportedContentPart,
    InvalidToolCall,
};

const EffortEntry = struct {
    name: []const u8,
    value: f64,
};

const effort_map = [_]EffortEntry{
    .{ .name = "none", .value = 0.0 },
    .{ .name = "minimal", .value = 0.1 },
    .{ .name = "low", .value = 0.2 },
    .{ .name = "medium", .value = 0.7 },
    .{ .name = "high", .value = 0.9 },
    .{ .name = "xhigh", .value = 0.99 },
    .{ .name = "max", .value = 0.99 },
};

fn roleToToken(role: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, role, "user")) return TOK_MSG_USER;
    if (std.mem.eql(u8, role, "assistant")) return TOK_MSG_MODEL;
    if (std.mem.eql(u8, role, "system")) return TOK_MSG_SYSTEM;
    if (std.mem.eql(u8, role, "tool")) return TOK_MSG_TOOL;
    return null;
}

fn isTruthy(val: ?std.json.Value) bool {
    const v = val orelse return false;
    return switch (v) {
        .null => false,
        .bool => |b| b,
        .integer => |i| i != 0,
        .float => |f| f != 0.0,
        .number_string => |s| s.len > 0 and !std.mem.eql(u8, s, "0"),
        .string => |s| s.len > 0,
        .array => |a| a.items.len > 0,
        .object => |o| o.count() > 0,
    };
}

/// Format an effort float in [0.0, 0.99] identically to Jinja/Python:
/// - 0.0 -> "0"
/// - >= 1e-4 -> shortest decimal (e.g. 0.9, 0.1, 0.99, 0.25)
/// - < 1e-4 -> Python scientific notation (e.g. 1e-05, 1.5e-05)
pub fn formatEffort(buf: []u8, num: f64) []const u8 {
    if (num == 0.0) {
        buf[0] = '0';
        return buf[0..1];
    }
    if (num >= 1e-4) {
        return std.fmt.bufPrint(buf, "{d}", .{num}) catch "0.9";
    }
    // Scientific notation matching Python str(float) for 0 < num < 1e-4
    var tmp: [64]u8 = undefined;
    const raw = std.fmt.bufPrint(&tmp, "{e}", .{num}) catch return "0";
    // Zig {e} outputs e.g. "1e-5" or "1.5e-5". Python pads negative exponents to at least 2 digits: "1e-05".
    const e_pos = std.mem.indexOfScalar(u8, raw, 'e') orelse {
        @memcpy(buf[0..raw.len], raw);
        return buf[0..raw.len];
    };
    const mantissa = raw[0..e_pos];
    const exp_part = raw[e_pos + 1 ..]; // e.g. "-5" or "-12"
    var idx: usize = 0;
    @memcpy(buf[idx .. idx + mantissa.len], mantissa);
    idx += mantissa.len;
    buf[idx] = 'e';
    idx += 1;
    if (exp_part.len >= 2 and (exp_part[0] == '-' or exp_part[0] == '+')) {
        buf[idx] = exp_part[0];
        idx += 1;
        const digits = exp_part[1..];
        if (digits.len == 1) {
            buf[idx] = '0';
            idx += 1;
        }
        @memcpy(buf[idx .. idx + digits.len], digits);
        idx += digits.len;
    } else {
        @memcpy(buf[idx .. idx + exp_part.len], exp_part);
        idx += exp_part.len;
    }
    return buf[0..idx];
}

fn writeJsonString(alloc: Allocator, out: *std.ArrayList(u8), s: []const u8) Allocator.Error!void {
    try out.append(alloc, '"');
    for (s) |c| {
        switch (c) {
            '"' => try out.appendSlice(alloc, "\\\""),
            '\\' => try out.appendSlice(alloc, "\\\\"),
            '\x08' => try out.appendSlice(alloc, "\\b"),
            '\x0c' => try out.appendSlice(alloc, "\\f"),
            '\n' => try out.appendSlice(alloc, "\\n"),
            '\r' => try out.appendSlice(alloc, "\\r"),
            '\t' => try out.appendSlice(alloc, "\\t"),
            0x00...0x07, 0x0b, 0x0e...0x1f => {
                var esc: [6]u8 = undefined;
                const slice = std.fmt.bufPrint(&esc, "\\u{x:0>4}", .{c}) catch unreachable;
                try out.appendSlice(alloc, slice);
            },
            else => try out.append(alloc, c),
        }
    }
    try out.append(alloc, '"');
}

fn writeCanonicalJson(alloc: Allocator, out: *std.ArrayList(u8), val: std.json.Value, sort_keys: bool) Allocator.Error!void {
    switch (val) {
        .null => try out.appendSlice(alloc, "null"),
        .bool => |b| try out.appendSlice(alloc, if (b) "true" else "false"),
        .integer => |i| try out.print(alloc, "{d}", .{i}),
        .float => |f| {
            if (std.math.isNan(f) or std.math.isInf(f)) {
                try out.appendSlice(alloc, "null");
            } else {
                var fbuf: [64]u8 = undefined;
                const s = std.fmt.bufPrint(&fbuf, "{d}", .{f}) catch "0";
                try out.appendSlice(alloc, s);
                // Match Python json.dumps float formatting: keep ".0" if whole number
                if (std.mem.indexOfAny(u8, s, ".eE") == null) {
                    try out.appendSlice(alloc, ".0");
                }
            }
        },
        .number_string => |ns| try out.appendSlice(alloc, ns),
        .string => |s| try writeJsonString(alloc, out, s),
        .array => |arr| {
            try out.append(alloc, '[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try out.append(alloc, ',');
                try writeCanonicalJson(alloc, out, item, sort_keys);
            }
            try out.append(alloc, ']');
        },
        .object => |obj| {
            try out.append(alloc, '{');
            if (sort_keys) {
                const count = obj.count();
                if (count > 0) {
                    const keys = try alloc.alloc([]const u8, count);
                    defer alloc.free(keys);
                    var it = obj.iterator();
                    var k: usize = 0;
                    while (it.next()) |entry| : (k += 1) {
                        keys[k] = entry.key_ptr.*;
                    }
                    std.mem.sort([]const u8, keys, {}, struct {
                        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                            return std.mem.order(u8, a, b) == .lt;
                        }
                    }.lessThan);
                    for (keys, 0..) |key, i| {
                        if (i > 0) try out.append(alloc, ',');
                        try writeJsonString(alloc, out, key);
                        try out.append(alloc, ':');
                        try writeCanonicalJson(alloc, out, obj.get(key).?, sort_keys);
                    }
                }
            } else {
                var it = obj.iterator();
                var i: usize = 0;
                while (it.next()) |entry| : (i += 1) {
                    if (i > 0) try out.append(alloc, ',');
                    try writeJsonString(alloc, out, entry.key_ptr.*);
                    try out.append(alloc, ':');
                    try writeCanonicalJson(alloc, out, entry.value_ptr.*, sort_keys);
                }
            }
            try out.append(alloc, '}');
        },
    }
}

fn pushSegment(alloc: Allocator, segs: *std.ArrayList(Segment), text: []const u8, parse_special: bool) Allocator.Error!void {
    if (text.len == 0) return;
    // Merge adjacent segments with the same parse_special flag to minimize tokenize calls
    // while strictly keeping parse_special=false isolated from parse_special=true.
    if (segs.items.len > 0 and segs.items[segs.items.len - 1].parse_special == parse_special) {
        const prev = segs.items[segs.items.len - 1].text;
        const merged = try alloc.alloc(u8, prev.len + text.len);
        @memcpy(merged[0..prev.len], prev);
        @memcpy(merged[prev.len..], text);
        segs.items[segs.items.len - 1].text = merged;
        return;
    }
    const duped = try alloc.dupe(u8, text);
    try segs.append(alloc, .{ .text = duped, .parse_special = parse_special });
}

fn setErr(err_out: ?*[]const u8, alloc: Allocator, comptime fmt: []const u8, args: anytype) void {
    if (err_out) |ptr| {
        ptr.* = std.fmt.allocPrint(alloc, fmt, args) catch "template error";
    }
}

fn emitThinkingEffort(
    alloc: Allocator,
    segs: *std.ArrayList(Segment),
    reasoning_effort: ?std.json.Value,
    err_out: ?*[]const u8,
) RenderError!void {
    var num: f64 = 0.9;
    if (reasoning_effort) |eff| {
        switch (eff) {
            .null => num = 0.9,
            .string => |s| {
                const trimmed = std.mem.trim(u8, s, " \t\r\n\x0b\x0c");
                var found: ?f64 = null;
                for (effort_map) |e| {
                    if (std.mem.eql(u8, trimmed, e.name)) {
                        found = e.value;
                        break;
                    }
                }
                if (found) |v| {
                    num = v;
                } else {
                    setErr(err_out, alloc, "Unknown reasoning_effort: {s}", .{s});
                    return error.UnknownReasoningEffort;
                }
            },
            .integer => |i| num = @floatFromInt(i),
            .float => |f| num = f,
            .number_string => |ns| {
                num = std.fmt.parseFloat(f64, ns) catch {
                    setErr(err_out, alloc, "Invalid reasoning_effort number: {s}", .{ns});
                    return error.UnknownReasoningEffort;
                };
            },
            else => {
                setErr(err_out, alloc, "reasoning_effort must be a string or number", .{});
                return error.UnknownReasoningEffort;
            },
        }
    }

    if (std.math.isNan(num) or num < 0.0 or num > 0.99) {
        setErr(err_out, alloc, "reasoning_effort must be in [0.0, 0.99]", .{});
        return error.ReasoningEffortOutOfRange;
    }

    var fbuf: [64]u8 = undefined;
    const formatted = formatEffort(&fbuf, num);

    try pushSegment(alloc, segs, TOK_MSG_SYSTEM, true);
    try pushSegment(alloc, segs, TOK_CONTENT_TEXT, true);
    var line_buf: [96]u8 = undefined;
    const line = std.fmt.bufPrint(&line_buf, "Thinking effort level: {s}", .{formatted}) catch unreachable;
    try pushSegment(alloc, segs, line, false);
    try pushSegment(alloc, segs, TOK_END_MESSAGE, true);
}

fn emitHeader(alloc: Allocator, segs: *std.ArrayList(Segment), role_tok: []const u8, header_name: []const u8) Allocator.Error!void {
    try pushSegment(alloc, segs, role_tok, true);
    if (header_name.len > 0) {
        try pushSegment(alloc, segs, header_name, false);
    }
}

/// Renders OpenAI chat messages, optional tools, and optional reasoning_effort into
/// segments `{ text, parse_special }` matching chat_template.jinja exactly.
/// Caller should use an arena allocator (segment slices are owned by `alloc`).
pub fn render(
    alloc: Allocator,
    messages_val: std.json.Value,
    tools_val: ?std.json.Value,
    reasoning_effort: ?std.json.Value,
    err_out: ?*[]const u8,
) RenderError![]Segment {
    if (messages_val != .array) {
        setErr(err_out, alloc, "messages must be an array", .{});
        return error.InvalidInput;
    }
    const messages = messages_val.array.items;

    var segs: std.ArrayList(Segment) = .empty;

    // 1. Tool declarations
    if (tools_val) |tv| {
        if (tv == .array and tv.array.items.len > 0) {
            var json_buf: std.ArrayList(u8) = .empty;
            try json_buf.append(alloc, '[');
            for (tv.array.items, 0..) |tool, idx| {
                if (tool != .object) {
                    setErr(err_out, alloc, "each tool must be an object", .{});
                    return error.InvalidToolCall;
                }
                if (idx > 0) try json_buf.append(alloc, ',');

                const fn_val = if (tool.object.get("function")) |f| f else tool;
                if (fn_val != .object) {
                    setErr(err_out, alloc, "tool function must be an object", .{});
                    return error.InvalidToolCall;
                }
                const fn_obj = fn_val.object;
                const name_val = fn_obj.get("name") orelse {
                    setErr(err_out, alloc, "tool function missing name", .{});
                    return error.InvalidToolCall;
                };
                if (name_val != .string) {
                    setErr(err_out, alloc, "tool function name must be a string", .{});
                    return error.InvalidToolCall;
                }

                const desc_val: std.json.Value = if (fn_obj.get("description")) |d|
                    (if (isTruthy(d)) d else .{ .string = "" })
                else
                    .{ .string = "" };

                const empty_obj: std.json.Value = .{ .object = .empty };
                const params_val: std.json.Value = if (fn_obj.get("parameters")) |p|
                    (if (isTruthy(p)) p else empty_obj)
                else
                    empty_obj;

                const type_val: std.json.Value = if (tool.object.get("type")) |t|
                    (if (isTruthy(t)) t else .{ .string = "function" })
                else
                    .{ .string = "function" };

                // Keys in exact alphabetical order (sort_keys=true): description, name, parameters, type
                try json_buf.appendSlice(alloc, "{\"description\":");
                try writeCanonicalJson(alloc, &json_buf, desc_val, true);
                try json_buf.appendSlice(alloc, ",\"name\":");
                try writeJsonString(alloc, &json_buf, name_val.string);
                try json_buf.appendSlice(alloc, ",\"parameters\":");
                try writeCanonicalJson(alloc, &json_buf, params_val, true);
                try json_buf.appendSlice(alloc, ",\"type\":");
                try writeCanonicalJson(alloc, &json_buf, type_val, true);
                try json_buf.append(alloc, '}');
            }
            try json_buf.append(alloc, ']');

            try pushSegment(alloc, &segs, TOK_MSG_SYSTEM, true);
            try pushSegment(alloc, &segs, "tool_declare", false);
            try pushSegment(alloc, &segs, TOK_CONTENT_XML, true);
            try pushSegment(alloc, &segs, json_buf.items, false);
            try pushSegment(alloc, &segs, TOK_END_MESSAGE, true);
        }
    }

    // 2. Messages loop
    var effort_emitted = false;
    for (messages) |msg_val| {
        if (msg_val != .object) {
            setErr(err_out, alloc, "each message must be an object", .{});
            return error.InvalidInput;
        }
        const msg = msg_val.object;
        const role_v = msg.get("role") orelse {
            setErr(err_out, alloc, "message missing role", .{});
            return error.InvalidInput;
        };
        if (role_v != .string) {
            setErr(err_out, alloc, "message role must be a string", .{});
            return error.InvalidInput;
        }
        const role = role_v.string;
        const rtok = roleToToken(role) orelse {
            setErr(err_out, alloc, "Unknown message role: {s}", .{role});
            return error.UnknownRole;
        };

        if (!effort_emitted and !std.mem.eql(u8, role, "system")) {
            try emitThinkingEffort(alloc, &segs, reasoning_effort, err_out);
            effort_emitted = true;
        }

        // Compute header suffix (name)
        var header_name: std.ArrayList(u8) = .empty;
        if (std.mem.eql(u8, role, "tool")) {
            const name_v = msg.get("name");
            if (name_v != null and name_v.? == .string and name_v.?.string.len > 0) {
                try header_name.appendSlice(alloc, name_v.?.string);
            } else if (msg.get("tool_call_id")) |tcid_v| {
                if (tcid_v == .string and tcid_v.string.len > 0) {
                    const target_id = tcid_v.string;
                    for (messages) |prev_v| {
                        if (prev_v != .object) continue;
                        const prev = prev_v.object;
                        const pr = prev.get("role") orelse continue;
                        if (pr != .string or !std.mem.eql(u8, pr.string, "assistant")) continue;
                        const ptcs = prev.get("tool_calls") orelse continue;
                        if (ptcs != .array) continue;
                        for (ptcs.array.items) |tc_v| {
                            if (tc_v != .object) continue;
                            const id_v = tc_v.object.get("id") orelse continue;
                            if (id_v != .string or !std.mem.eql(u8, id_v.string, target_id)) continue;
                            const fn_v = tc_v.object.get("function") orelse continue;
                            if (fn_v != .object) continue;
                            const fn_name = fn_v.object.get("name") orelse continue;
                            if (fn_name == .string) {
                                try header_name.appendSlice(alloc, fn_name.string);
                            }
                        }
                    }
                }
            }
        } else if (std.mem.eql(u8, role, "assistant")) {
            if (msg.get("name")) |nv| {
                if (nv == .string and nv.string.len > 0) {
                    try header_name.appendSlice(alloc, nv.string);
                }
            }
        }

        const is_assistant = std.mem.eql(u8, role, "assistant");
        const rc_val = msg.get("reasoning_content");
        const has_reasoning = is_assistant and rc_val != null and rc_val.? == .string and rc_val.?.string.len > 0;
        const tc_val = msg.get("tool_calls");
        const has_tool_calls = is_assistant and tc_val != null and tc_val.? == .array and tc_val.?.array.items.len > 0;

        // Assistant reasoning_content
        if (has_reasoning) {
            try emitHeader(alloc, &segs, rtok, header_name.items);
            try pushSegment(alloc, &segs, TOK_CONTENT_THINKING, true);
            try pushSegment(alloc, &segs, rc_val.?.string, false);
            try pushSegment(alloc, &segs, TOK_END_MESSAGE, true);
        }

        // Message content
        if (msg.get("content")) |content_v| {
            switch (content_v) {
                .string => |s| {
                    if (s.len > 0 or !(is_assistant and (has_tool_calls or has_reasoning))) {
                        try emitHeader(alloc, &segs, rtok, header_name.items);
                        try pushSegment(alloc, &segs, TOK_CONTENT_TEXT, true);
                        try pushSegment(alloc, &segs, s, false);
                        try pushSegment(alloc, &segs, TOK_END_MESSAGE, true);
                    }
                },
                .array => |arr| {
                    for (arr.items) |part| {
                        switch (part) {
                            .string => |ps| {
                                try emitHeader(alloc, &segs, rtok, header_name.items);
                                try pushSegment(alloc, &segs, TOK_CONTENT_TEXT, true);
                                try pushSegment(alloc, &segs, ps, false);
                                try pushSegment(alloc, &segs, TOK_END_MESSAGE, true);
                            },
                            .object => |po| {
                                const ptype: ?[]const u8 = if (po.get("type")) |tv|
                                    (if (tv == .string) tv.string else null)
                                else
                                    null;

                                if (ptype == null or std.mem.eql(u8, ptype.?, "text") or std.mem.eql(u8, ptype.?, "input_text")) {
                                    const text_part: []const u8 = if (po.get("text")) |tx|
                                        (if (tx == .string) tx.string else "")
                                    else
                                        "";
                                    try emitHeader(alloc, &segs, rtok, header_name.items);
                                    try pushSegment(alloc, &segs, TOK_CONTENT_TEXT, true);
                                    try pushSegment(alloc, &segs, text_part, false);
                                    try pushSegment(alloc, &segs, TOK_END_MESSAGE, true);
                                } else if (std.mem.eql(u8, ptype.?, "image") or
                                    std.mem.eql(u8, ptype.?, "input_image") or
                                    std.mem.eql(u8, ptype.?, "image_url"))
                                {
                                    setErr(err_out, alloc, "images not supported", .{});
                                    return error.ImagesNotSupported;
                                } else if (std.mem.eql(u8, ptype.?, "audio") or
                                    std.mem.eql(u8, ptype.?, "input_audio") or
                                    std.mem.eql(u8, ptype.?, "audio_url"))
                                {
                                    setErr(err_out, alloc, "Inkling 2 does not accept audio input", .{});
                                    return error.AudioNotSupported;
                                } else {
                                    setErr(err_out, alloc, "Unsupported content part type: {s}", .{ptype.?});
                                    return error.UnsupportedContentPart;
                                }
                            },
                            else => {
                                setErr(err_out, alloc, "Unsupported content part", .{});
                                return error.UnsupportedContentPart;
                            },
                        }
                    }
                },
                .null => {},
                else => {
                    setErr(err_out, alloc, "message content must be a string, array, or null", .{});
                    return error.InvalidInput;
                },
            }
        }

        // Assistant tool_calls
        if (has_tool_calls) {
            for (tc_val.?.array.items) |tc_item| {
                if (tc_item != .object) {
                    setErr(err_out, alloc, "tool_call must be an object", .{});
                    return error.InvalidToolCall;
                }
                const fn_v = tc_item.object.get("function") orelse {
                    setErr(err_out, alloc, "tool_call missing function", .{});
                    return error.InvalidToolCall;
                };
                if (fn_v != .object) {
                    setErr(err_out, alloc, "tool_call function must be an object", .{});
                    return error.InvalidToolCall;
                }
                const fn_name_v = fn_v.object.get("name") orelse {
                    setErr(err_out, alloc, "tool call function name must be a string", .{});
                    return error.InvalidToolCall;
                };
                if (fn_name_v != .string) {
                    setErr(err_out, alloc, "tool call function name must be a string", .{});
                    return error.InvalidToolCall;
                }
                const fn_name = fn_name_v.string;
                if (header_name.items.len > 0 and !std.mem.eql(u8, header_name.items, fn_name)) {
                    setErr(err_out, alloc, "assistant name {s} does not match its tool call {s}", .{ header_name.items, fn_name });
                    return error.InvalidToolCall;
                }

                // Canonicalize arguments: OpenAI clients pass JSON string, or already parsed object
                var args_obj: std.json.ObjectMap = .empty;
                if (fn_v.object.get("arguments")) |av| {
                    switch (av) {
                        .null => {},
                        .object => |o| args_obj = o,
                        .string => |astr| {
                            const trimmed = std.mem.trim(u8, astr, " \t\r\n");
                            if (trimmed.len > 0) {
                                const parsed = std.json.parseFromSliceLeaky(std.json.Value, alloc, trimmed, .{}) catch {
                                    setErr(err_out, alloc, "tool call arguments string is not valid JSON", .{});
                                    return error.InvalidToolCall;
                                };
                                if (parsed != .object) {
                                    setErr(err_out, alloc, "tool call arguments must be an object", .{});
                                    return error.InvalidToolCall;
                                }
                                args_obj = parsed.object;
                            }
                        },
                        else => {
                            setErr(err_out, alloc, "tool call arguments must be an object", .{});
                            return error.InvalidToolCall;
                        },
                    }
                }

                var call_json: std.ArrayList(u8) = .empty;
                try call_json.appendSlice(alloc, "{\"name\":");
                try writeJsonString(alloc, &call_json, fn_name);
                try call_json.appendSlice(alloc, ",\"args\":{");

                const key_count = args_obj.count();
                if (key_count > 0) {
                    const keys = try alloc.alloc([]const u8, key_count);
                    defer alloc.free(keys);
                    var it = args_obj.iterator();
                    var k: usize = 0;
                    while (it.next()) |entry| : (k += 1) {
                        keys[k] = entry.key_ptr.*;
                    }
                    std.mem.sort([]const u8, keys, {}, struct {
                        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                            return std.mem.order(u8, a, b) == .lt;
                        }
                    }.lessThan);
                    for (keys, 0..) |key, i| {
                        if (i > 0) try call_json.append(alloc, ',');
                        try writeJsonString(alloc, &call_json, key);
                        try call_json.append(alloc, ':');
                        try writeCanonicalJson(alloc, &call_json, args_obj.get(key).?, false);
                    }
                }
                try call_json.appendSlice(alloc, "}}");

                try pushSegment(alloc, &segs, TOK_MSG_MODEL, true);
                try pushSegment(alloc, &segs, fn_name, false);
                try pushSegment(alloc, &segs, TOK_CONTENT_TOOL_JSON, true);
                try pushSegment(alloc, &segs, call_json.items, false);
                try pushSegment(alloc, &segs, TOK_END_MESSAGE, true);
            }
        }

        // End of assistant turn
        if (is_assistant) {
            try pushSegment(alloc, &segs, TOK_END_SAMPLING, true);
        }
    }

    // 3. Emit thinking effort if no non-system message triggered it
    if (!effort_emitted) {
        try emitThinkingEffort(alloc, &segs, reasoning_effort, err_out);
    }

    return segs.toOwnedSlice(alloc);
}

/// Concatenate segments into a single flat string (useful for testing/debugging).
pub fn segmentsToString(alloc: Allocator, segs: []const Segment) Allocator.Error![]u8 {
    var total: usize = 0;
    for (segs) |s| total += s.text.len;
    const out = try alloc.alloc(u8, total);
    var pos: usize = 0;
    for (segs) |s| {
        @memcpy(out[pos .. pos + s.text.len], s.text);
        pos += s.text.len;
    }
    return out;
}

// ============================================================================
// UTF-8 incomplete sequence helper
// ============================================================================

/// Returns the number of trailing bytes in `data` that form an incomplete
/// multi-byte UTF-8 sequence (0..3). Complete sequences and invalid bytes return 0.
pub fn incompleteUtf8TailLen(data: []const u8) usize {
    if (data.len == 0) return 0;
    const max_back = @min(data.len, 3);
    var back: usize = 1;
    while (back <= max_back) : (back += 1) {
        const b = data[data.len - back];
        if (b < 0x80) return 0; // ASCII terminates any trailing sequence
        if ((b & 0xC0) == 0x80) continue; // continuation byte, keep walking back
        // Found a leading byte at data.len - back
        const expected: usize = if (b >= 0xC2 and b <= 0xDF)
            2
        else if (b >= 0xE0 and b <= 0xEF)
            3
        else if (b >= 0xF0 and b <= 0xF4)
            4
        else
            return 0; // invalid leading byte, cannot be completed
        if (back < expected) return back;
        return 0; // already complete (or overlong)
    }
    return 0;
}

/// Returns the longest suffix length of `data` that is a proper prefix of any stop string.
pub fn longestStopPrefixAtEnd(data: []const u8, stops: []const []const u8) usize {
    var best: usize = 0;
    for (stops) |stop| {
        if (stop.len <= 1) continue;
        const max_k = @min(data.len, stop.len - 1);
        var k: usize = max_k;
        while (k > best) : (k -= 1) {
            if (std.mem.eql(u8, data[data.len - k ..], stop[0..k])) {
                best = k;
                break;
            }
        }
    }
    return best;
}

/// Finds the earliest occurrence of any non-empty stop string in `data`.
pub fn findEarliestStop(data: []const u8, stops: []const []const u8) ?usize {
    var earliest: ?usize = null;
    for (stops) |stop| {
        if (stop.len == 0) continue;
        if (std.mem.indexOf(u8, data, stop)) |pos| {
            if (earliest == null or pos < earliest.?) {
                earliest = pos;
            }
        }
    }
    return earliest;
}

// ============================================================================
// Inkling Output Parser (uses engine.special() token IDs)
// ============================================================================

pub const SpecialIds = struct {
    message_model: ?u32 = null,
    content_thinking: ?u32 = null,
    content_text: ?u32 = null,
    content_tool_json: ?u32 = null,
    end_message: ?u32 = null,
    end_sampling: ?u32 = null,
    // Other specials to suppress if ever emitted
    message_user: ?u32 = null,
    message_system: ?u32 = null,
    message_tool: ?u32 = null,
    content_xml: ?u32 = null,
    content_image: ?u32 = null,

    pub fn isOtherSpecial(self: SpecialIds, id: u32) bool {
        const others = [_]?u32{
            self.message_user,
            self.message_system,
            self.message_tool,
            self.content_xml,
            self.content_image,
        };
        for (others) |o| {
            if (o != null and o.? == id) return true;
        }
        return false;
    }
};

pub const ParsedToolCall = struct {
    index: u32,
    id: []u8,
    name: []u8,
    arguments: []u8,
};

pub const OutputEvent = union(enum) {
    reasoning: []const u8,
    content: []const u8,
    tool_call: ParsedToolCall,
    stopped_by_stop_string,
    stopped_by_eog,
};

pub const OutputParser = struct {
    pub const State = enum {
        idle,
        in_header,
        in_thinking,
        in_text,
        in_tool_json,
        done,
    };

    alloc: Allocator,
    specials: SpecialIds,
    stops: []const []const u8,
    state: State = .idle,
    seen_any_tag: bool = false,
    tool_count: u32 = 0,
    thinking_buf: std.ArrayList(u8) = .empty,
    text_buf: std.ArrayList(u8) = .empty,
    header_buf: std.ArrayList(u8) = .empty,
    tool_buf: std.ArrayList(u8) = .empty,

    pub fn init(alloc: Allocator, specials: SpecialIds, stops: []const []const u8) OutputParser {
        return .{
            .alloc = alloc,
            .specials = specials,
            .stops = stops,
        };
    }

    pub fn deinit(self: *OutputParser) void {
        self.thinking_buf.deinit(self.alloc);
        self.text_buf.deinit(self.alloc);
        self.header_buf.deinit(self.alloc);
        self.tool_buf.deinit(self.alloc);
    }

    fn flushThinking(self: *OutputParser, force_all: bool, events: *std.ArrayList(OutputEvent)) Allocator.Error!void {
        if (self.thinking_buf.items.len == 0) return;
        const hold = if (force_all) 0 else incompleteUtf8TailLen(self.thinking_buf.items);
        const emit_len = self.thinking_buf.items.len - hold;
        if (emit_len > 0) {
            const slice = try self.alloc.dupe(u8, self.thinking_buf.items[0..emit_len]);
            try events.append(self.alloc, .{ .reasoning = slice });
            if (hold > 0) {
                std.mem.copyForwards(u8, self.thinking_buf.items[0..hold], self.thinking_buf.items[emit_len..]);
            }
            self.thinking_buf.items.len = hold;
        }
    }

    fn flushText(self: *OutputParser, force_all: bool, events: *std.ArrayList(OutputEvent)) Allocator.Error!bool {
        if (self.text_buf.items.len == 0) return false;

        // Check for stop strings first
        if (self.stops.len > 0) {
            if (findEarliestStop(self.text_buf.items, self.stops)) |stop_pos| {
                if (stop_pos > 0) {
                    const slice = try self.alloc.dupe(u8, self.text_buf.items[0..stop_pos]);
                    try events.append(self.alloc, .{ .content = slice });
                }
                self.text_buf.clearRetainingCapacity();
                self.state = .done;
                try events.append(self.alloc, .stopped_by_stop_string);
                return true;
            }
        }

        var hold: usize = 0;
        if (!force_all) {
            const stop_hold = longestStopPrefixAtEnd(self.text_buf.items, self.stops);
            const prefix_len = self.text_buf.items.len - stop_hold;
            const utf8_hold = incompleteUtf8TailLen(self.text_buf.items[0..prefix_len]);
            hold = stop_hold + utf8_hold;
        }

        const emit_len = self.text_buf.items.len - hold;
        if (emit_len > 0) {
            const slice = try self.alloc.dupe(u8, self.text_buf.items[0..emit_len]);
            try events.append(self.alloc, .{ .content = slice });
            if (hold > 0) {
                std.mem.copyForwards(u8, self.text_buf.items[0..hold], self.text_buf.items[emit_len..]);
            }
            self.text_buf.items.len = hold;
        }
        return false;
    }

    fn finalizeToolCall(self: *OutputParser, events: *std.ArrayList(OutputEvent)) Allocator.Error!void {
        const raw = std.mem.trim(u8, self.tool_buf.items, " \t\r\n");
        if (raw.len == 0) {
            self.tool_buf.clearRetainingCapacity();
            self.header_buf.clearRetainingCapacity();
            return;
        }

        var name: []const u8 = std.mem.trim(u8, self.header_buf.items, " \t\r\n");
        var args_str: []u8 = undefined;
        var args_owned = false;

        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        const aa = arena.allocator();

        if (std.json.parseFromSliceLeaky(std.json.Value, aa, raw, .{})) |val| {
            if (val == .object) {
                if (val.object.get("name")) |nv| {
                    if (nv == .string) name = nv.string;
                }
                if (val.object.get("args") orelse val.object.get("arguments")) |av| {
                    var abuf: std.ArrayList(u8) = .empty;
                    if (av == .string) {
                        try abuf.appendSlice(self.alloc, av.string);
                    } else {
                        try writeCanonicalJson(self.alloc, &abuf, av, false);
                    }
                    args_str = try abuf.toOwnedSlice(self.alloc);
                    args_owned = true;
                }
            }
        } else |_| {}

        if (!args_owned) {
            args_str = try self.alloc.dupe(u8, "{}");
        }

        const idx = self.tool_count;
        self.tool_count += 1;
        const id = try std.fmt.allocPrint(self.alloc, "call_{d}", .{idx});
        const name_dup = try self.alloc.dupe(u8, name);

        try events.append(self.alloc, .{
            .tool_call = .{
                .index = idx,
                .id = id,
                .name = name_dup,
                .arguments = args_str,
            },
        });

        self.tool_buf.clearRetainingCapacity();
        self.header_buf.clearRetainingCapacity();
    }

    fn flushCurrentBlock(self: *OutputParser, events: *std.ArrayList(OutputEvent)) Allocator.Error!bool {
        switch (self.state) {
            .in_thinking => {
                try self.flushThinking(true, events);
                return false;
            },
            .in_text => {
                return try self.flushText(true, events);
            },
            .in_tool_json => {
                try self.finalizeToolCall(events);
                return false;
            },
            else => return false,
        }
    }

    /// Feeds one generated token `(id, bytes)` into the parser and appends resulting events.
    /// Caller owns any slices in returned events (allocated from `self.alloc`).
    pub fn feed(self: *OutputParser, id: u32, bytes: []const u8, is_eog: bool, events: *std.ArrayList(OutputEvent)) Allocator.Error!void {
        if (self.state == .done) return;

        // End of sampling or engine EOG
        if (is_eog or (self.specials.end_sampling != null and id == self.specials.end_sampling.?)) {
            self.seen_any_tag = true;
            if (try self.flushCurrentBlock(events)) return;
            self.state = .done;
            try events.append(self.alloc, .stopped_by_eog);
            return;
        }

        if (self.specials.message_model != null and id == self.specials.message_model.?) {
            self.seen_any_tag = true;
            if (try self.flushCurrentBlock(events)) return;
            self.header_buf.clearRetainingCapacity();
            self.state = .in_header;
            return;
        }

        if (self.specials.content_thinking != null and id == self.specials.content_thinking.?) {
            self.seen_any_tag = true;
            if (self.state != .in_header and self.state != .idle) {
                if (try self.flushCurrentBlock(events)) return;
            }
            self.header_buf.clearRetainingCapacity();
            self.state = .in_thinking;
            return;
        }

        if (self.specials.content_text != null and id == self.specials.content_text.?) {
            self.seen_any_tag = true;
            if (self.state != .in_header and self.state != .idle) {
                if (try self.flushCurrentBlock(events)) return;
            }
            self.header_buf.clearRetainingCapacity();
            self.state = .in_text;
            return;
        }

        if (self.specials.content_tool_json != null and id == self.specials.content_tool_json.?) {
            self.seen_any_tag = true;
            if (self.state != .in_header and self.state != .idle) {
                if (try self.flushCurrentBlock(events)) return;
            }
            self.tool_buf.clearRetainingCapacity();
            self.state = .in_tool_json;
            return;
        }

        if (self.specials.end_message != null and id == self.specials.end_message.?) {
            self.seen_any_tag = true;
            if (try self.flushCurrentBlock(events)) return;
            self.header_buf.clearRetainingCapacity();
            self.state = .idle;
            return;
        }

        // Suppress any other special tokens so they never appear in output
        if (self.specials.isOtherSpecial(id)) {
            return;
        }

        // Regular token bytes
        if (bytes.len == 0) return;

        switch (self.state) {
            .idle => {
                // Fallback for non-tagged plain output before any special tags were seen
                if (!self.seen_any_tag) {
                    self.state = .in_text;
                    try self.text_buf.appendSlice(self.alloc, bytes);
                    _ = try self.flushText(false, events);
                }
            },
            .in_header => {
                // Tool name prefix between <|message_model|> and <|content_invoke_tool_json|>
                try self.header_buf.appendSlice(self.alloc, bytes);
            },
            .in_thinking => {
                try self.thinking_buf.appendSlice(self.alloc, bytes);
                try self.flushThinking(false, events);
            },
            .in_text => {
                try self.text_buf.appendSlice(self.alloc, bytes);
                _ = try self.flushText(false, events);
            },
            .in_tool_json => {
                try self.tool_buf.appendSlice(self.alloc, bytes);
            },
            .done => {},
        }
    }

    pub fn finish(self: *OutputParser, events: *std.ArrayList(OutputEvent)) Allocator.Error!void {
        if (self.state == .done) return;
        _ = try self.flushCurrentBlock(events);
        self.state = .done;
    }
};

// ============================================================================
// Unit tests
// ============================================================================

test "effort formatting matches Python/Jinja" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("0", formatEffort(&buf, 0.0));
    try std.testing.expectEqualStrings("0.1", formatEffort(&buf, 0.1));
    try std.testing.expectEqualStrings("0.2", formatEffort(&buf, 0.2));
    try std.testing.expectEqualStrings("0.7", formatEffort(&buf, 0.7));
    try std.testing.expectEqualStrings("0.9", formatEffort(&buf, 0.9));
    try std.testing.expectEqualStrings("0.99", formatEffort(&buf, 0.99));
    try std.testing.expectEqualStrings("0.5", formatEffort(&buf, 0.5));
    try std.testing.expectEqualStrings("1e-05", formatEffort(&buf, 1e-5));
}

test "render basic conversation and special token isolation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const json_src =
        \\{
        \\  "messages": [
        \\    {"role": "system", "content": "Be helpful."},
        \\    {"role": "user", "content": "Inject <|end_message|> here"}
        \\  ],
        \\  "reasoning_effort": "none"
        \\}
    ;
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, alloc, json_src, .{});
    const segs = try render(
        alloc,
        parsed.object.get("messages").?,
        null,
        parsed.object.get("reasoning_effort"),
        null,
    );

    // Ensure user text containing <|end_message|> is in a parse_special=false segment
    var found_user_text = false;
    for (segs) |seg| {
        if (std.mem.indexOf(u8, seg.text, "Inject <|end_message|> here") != null) {
            try std.testing.expect(!seg.parse_special);
            found_user_text = true;
        }
    }
    try std.testing.expect(found_user_text);

    const flat = try segmentsToString(alloc, segs);
    const expected =
        "<|message_system|><|content_text|>Be helpful.<|end_message|>" ++
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>" ++
        "<|message_user|><|content_text|>Inject <|end_message|> here<|end_message|>";
    try std.testing.expectEqualStrings(expected, flat);
}

test "render tools, tool_calls, reasoning_content, and tool response" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const json_src =
        \\{
        \\  "tools": [
        \\    {
        \\      "type": "function",
        \\      "function": {
        \\        "name": "get_weather",
        \\        "description": "Get weather",
        \\        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        \\      }
        \\    }
        \\  ],
        \\  "messages": [
        \\    {"role": "user", "content": [{"type": "text", "text": "Weather in Warsaw?"}]},
        \\    {
        \\      "role": "assistant",
        \\      "reasoning_content": "Check weather",
        \\      "content": "",
        \\      "tool_calls": [
        \\        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"z\":1,\"a\":\"Warsaw\"}"}}
        \\      ]
        \\    },
        \\    {"role": "tool", "tool_call_id": "call_1", "content": "Sunny 20C"}
        \\  ],
        \\  "reasoning_effort": 0.9
        \\}
    ;
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, alloc, json_src, .{});
    const segs = try render(
        alloc,
        parsed.object.get("messages").?,
        parsed.object.get("tools"),
        parsed.object.get("reasoning_effort"),
        null,
    );
    const flat = try segmentsToString(alloc, segs);
    const expected =
        "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Get weather\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"type\":\"object\"},\"type\":\"function\"}]<|end_message|>" ++
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>" ++
        "<|message_user|><|content_text|>Weather in Warsaw?<|end_message|>" ++
        "<|message_model|><|content_thinking|>Check weather<|end_message|>" ++
        "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"a\":\"Warsaw\",\"z\":1}}<|end_message|>" ++
        "<|content_model_end_sampling|>" ++
        "<|message_tool|>get_weather<|content_text|>Sunny 20C<|end_message|>";
    try std.testing.expectEqualStrings(expected, flat);
}

test "render rejects image parts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const json_src =
        \\{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}
    ;
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, alloc, json_src, .{});
    var err_msg: []const u8 = "";
    const res = render(alloc, parsed.object.get("messages").?, null, null, &err_msg);
    try std.testing.expectError(error.ImagesNotSupported, res);
    try std.testing.expectEqualStrings("images not supported", err_msg);
}

test "output parser splits thinking, text, utf-8, stop strings, and tool calls" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const specials: SpecialIds = .{
        .message_model = 1001,
        .content_thinking = 1004,
        .content_text = 1005,
        .content_tool_json = 1008,
        .end_message = 1009,
        .end_sampling = 1010,
    };

    var parser = OutputParser.init(alloc, specials, &.{});
    defer parser.deinit();

    var events: std.ArrayList(OutputEvent) = .empty;

    // Thinking + split UTF-8 ("ł" = 0xC5 0x82) + tool call
    try parser.feed(1001, "<|message_model|>", false, &events);
    try parser.feed(1004, "<|content_thinking|>", false, &events);
    try parser.feed(1, "My\xc5", false, &events);
    try std.testing.expectEqual(@as(usize, 1), events.items.len);
    try std.testing.expectEqualStrings("My", events.items[0].reasoning);
    try parser.feed(2, "\x82", false, &events);
    try std.testing.expectEqual(@as(usize, 2), events.items.len);
    try std.testing.expectEqualStrings("ł", events.items[1].reasoning);
    try parser.feed(1009, "<|end_message|>", false, &events);

    try parser.feed(1001, "<|message_model|>", false, &events);
    try parser.feed(3, "get_weather", false, &events);
    try parser.feed(1008, "<|content_invoke_tool_json|>", false, &events);
    try parser.feed(4, "{\"name\":\"get_weather\",\"args\":{\"city\":\"Warsaw\"}}", false, &events);
    try parser.feed(1009, "<|end_message|>", false, &events);
    try parser.feed(1010, "<|content_model_end_sampling|>", true, &events);

    try std.testing.expectEqual(@as(usize, 4), events.items.len);
    try std.testing.expectEqualStrings("get_weather", events.items[2].tool_call.name);
    try std.testing.expectEqualStrings("{\"city\":\"Warsaw\"}", events.items[2].tool_call.arguments);
    try std.testing.expect(events.items[3] == .stopped_by_eog);
}
