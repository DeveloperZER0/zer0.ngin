const std = @import("std");
const Allocator = std.mem.Allocator;
const chat_template = @import("chat_template.zig");

const UI_HTML = @embedFile("ui.html");
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 64 * 1024;

// ============================================================================
// Public Engine & Server API
// ============================================================================

pub const GenParams = struct {
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    seed: u64,
    stop: []const []const u8,
    cache_prompt: bool,
};

pub const GenStats = struct {
    prompt_tokens: u32 = 0,
    cached_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    prompt_ms: f64 = 0,
    decode_ms: f64 = 0,
};

pub const FinishReason = enum { stop, length, aborted };

pub const Engine = struct {
    ctx: *anyopaque,
    tokenize: *const fn (ctx: *anyopaque, alloc: std.mem.Allocator, text: []const u8, parse_special: bool) anyerror![]u32,
    // blocks; calls on_token(user, token_id, bytes) for each generated token; on_token returns false to abort. Returns finish reason.
    generate: *const fn (ctx: *anyopaque, prompt: []const u32, params: GenParams, stats: *GenStats, user: *anyopaque, on_token: *const fn (user: *anyopaque, id: u32, bytes: []const u8) bool) anyerror!FinishReason,
    special: *const fn (ctx: *anyopaque, literal: []const u8) ?u32,
    is_eog: *const fn (ctx: *anyopaque, id: u32) bool,
    n_ctx: u32,
};

// ============================================================================
// Winsock2 & Kernel32 FFI
// ============================================================================

const SOCKET = usize;
const INVALID_SOCKET: SOCKET = ~@as(usize, 0);
const SOCKET_ERROR: c_int = -1;
const AF_INET: c_int = 2;
const SOCK_STREAM: c_int = 1;
const IPPROTO_TCP: c_int = 6;
const SOL_SOCKET: c_int = 0xffff;
const SO_REUSEADDR: c_int = 0x0004;
const SO_SNDTIMEO: c_int = 0x1005;
const SO_RCVTIMEO: c_int = 0x1006;
const TCP_NODELAY: c_int = 0x0001;
const MSG_PEEK: c_int = 0x2;
const SD_BOTH: c_int = 2;

const SockAddrIn = extern struct {
    sin_family: u16,
    sin_port: u16,
    sin_addr: u32,
    sin_zero: [8]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
};

const FdSet = extern struct {
    fd_count: u32,
    fd_array: [64]SOCKET,
};

const TimeVal = extern struct {
    tv_sec: i32,
    tv_usec: i32,
};

const WsaDataBuf = extern struct {
    raw: [512]u8 align(8),
};

extern "ws2_32" fn WSAStartup(wVersionRequested: u16, lpWSAData: *WsaDataBuf) callconv(.winapi) c_int;
extern "ws2_32" fn WSACleanup() callconv(.winapi) c_int;
extern "ws2_32" fn WSAGetLastError() callconv(.winapi) c_int;
extern "ws2_32" fn socket(af: c_int, sock_type: c_int, protocol: c_int) callconv(.winapi) SOCKET;
extern "ws2_32" fn setsockopt(s: SOCKET, level: c_int, optname: c_int, optval: ?*const anyopaque, optlen: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn bind(s: SOCKET, name: *const anyopaque, namelen: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn listen(s: SOCKET, backlog: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn accept(s: SOCKET, addr: ?*anyopaque, addrlen: ?*c_int) callconv(.winapi) SOCKET;
extern "ws2_32" fn recv(s: SOCKET, buf: [*]u8, len: c_int, flags: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn send(s: SOCKET, buf: [*]const u8, len: c_int, flags: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn shutdown(s: SOCKET, how: c_int) callconv(.winapi) c_int;
extern "ws2_32" fn closesocket(s: SOCKET) callconv(.winapi) c_int;
extern "ws2_32" fn select(nfds: c_int, readfds: ?*FdSet, writefds: ?*FdSet, exceptfds: ?*FdSet, timeout: ?*const TimeVal) callconv(.winapi) c_int;

const SRWLOCK = extern struct {
    ptr: ?*anyopaque = null,
};

const CONDITION_VARIABLE = extern struct {
    ptr: ?*anyopaque = null,
};

const FILETIME = extern struct {
    dwLowDateTime: u32,
    dwHighDateTime: u32,
};

extern "kernel32" fn AcquireSRWLockExclusive(lock: *SRWLOCK) callconv(.winapi) void;
extern "kernel32" fn ReleaseSRWLockExclusive(lock: *SRWLOCK) callconv(.winapi) void;
extern "kernel32" fn SleepConditionVariableSRW(cv: *CONDITION_VARIABLE, lock: *SRWLOCK, dwMilliseconds: u32, flags: u32) callconv(.winapi) i32;
extern "kernel32" fn WakeConditionVariable(cv: *CONDITION_VARIABLE) callconv(.winapi) void;
extern "kernel32" fn GetSystemTimeAsFileTime(ft: *FILETIME) callconv(.winapi) void;
extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.winapi) i32;
extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.winapi) i32;

fn unixTimestampSeconds() i64 {
    var ft: FILETIME = undefined;
    GetSystemTimeAsFileTime(&ft);
    const ticks: u64 = (@as(u64, ft.dwHighDateTime) << 32) | @as(u64, ft.dwLowDateTime);
    // Windows FILETIME epoch is 1601-01-01; Unix epoch is 1970-01-01 (11644473600 seconds)
    const unix_offset_ticks: u64 = 11644473600 * 10_000_000;
    if (ticks < unix_offset_ticks) return 0;
    return @intCast((ticks - unix_offset_ticks) / 10_000_000);
}

fn perfCounterMs(start: i64, end: i64, freq: i64) f64 {
    if (freq <= 0 or end <= start) return 0.0;
    return @as(f64, @floatFromInt(end - start)) * 1000.0 / @as(f64, @floatFromInt(freq));
}

fn parseIpv4Host(host_in: []const u8) !u32 {
    const host = if (host_in.len == 0 or std.mem.eql(u8, host_in, "localhost"))
        "127.0.0.1"
    else
        host_in;

    var octets: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, host, '.');
    var idx: usize = 0;
    while (it.next()) |part| : (idx += 1) {
        if (idx >= 4 or part.len == 0) return error.InvalidHost;
        octets[idx] = std.fmt.parseInt(u8, part, 10) catch return error.InvalidHost;
    }
    if (idx != 4) return error.InvalidHost;
    // Network byte order (in_addr in little-endian memory is [a, b, c, d])
    return @bitCast(octets);
}

fn htons(port: u16) u16 {
    return std.mem.nativeToBig(u16, port);
}

fn isSocketDisconnected(sock: SOCKET) bool {
    var rfds: FdSet = .{ .fd_count = 1, .fd_array = undefined };
    rfds.fd_array[0] = sock;
    var efds: FdSet = .{ .fd_count = 1, .fd_array = undefined };
    efds.fd_array[0] = sock;
    const tv: TimeVal = .{ .tv_sec = 0, .tv_usec = 0 };

    const ready = select(0, &rfds, null, &efds, &tv);
    if (ready < 0) return true;
    if (ready == 0) return false;
    if (efds.fd_count > 0) return true;
    if (rfds.fd_count > 0) {
        var peek_byte: [1]u8 = undefined;
        const n = recv(sock, &peek_byte, 1, MSG_PEEK);
        // n == 0 means peer sent FIN (graceful close); n < 0 means error/RST
        if (n <= 0) return true;
    }
    return false;
}

fn sendAll(sock: SOCKET, data: []const u8) bool {
    var offset: usize = 0;
    while (offset < data.len) {
        const remaining = data.len - offset;
        const chunk_len: c_int = @intCast(@min(remaining, @as(usize, 1024 * 1024)));
        const n = send(sock, data.ptr + offset, chunk_len, 0);
        if (n <= 0) return false;
        offset += @intCast(n);
    }
    return true;
}

fn sendChunk(sock: SOCKET, payload: []const u8) bool {
    if (payload.len == 0) return true;
    var hdr_buf: [32]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "{x}\r\n", .{payload.len}) catch return false;
    if (!sendAll(sock, hdr)) return false;
    if (!sendAll(sock, payload)) return false;
    if (!sendAll(sock, "\r\n")) return false;
    return true;
}

fn sendChunkEnd(sock: SOCKET) bool {
    return sendAll(sock, "0\r\n\r\n");
}

// ============================================================================
// CORS validation (strictly http://127.0.0.1:* and http://localhost:*)
// ============================================================================

pub fn isAllowedOrigin(origin: []const u8) bool {
    const prefixes = [_][]const u8{
        "http://127.0.0.1",
        "http://localhost",
    };
    for (prefixes) |prefix| {
        if (!std.mem.startsWith(u8, origin, prefix)) continue;
        const rest = origin[prefix.len..];
        if (rest.len == 0) return true;
        if (rest[0] != ':') continue;
        const port_str = rest[1..];
        if (port_str.len == 0 or port_str.len > 5) return false;
        for (port_str) |c| {
            if (c < '0' or c > '9') return false;
        }
        const port = std.fmt.parseInt(u32, port_str, 10) catch return false;
        return port >= 1 and port <= 65535;
    }
    return false;
}

// ============================================================================
// Shared server state & single-slot generation gate
// ============================================================================

const ServerState = struct {
    alloc: Allocator,
    engine: *Engine,
    model_name: []const u8,
    created_ts: i64,
    qpc_freq: i64,
    req_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(1),
    gen_lock: SRWLOCK = .{},
    gen_cv: CONDITION_VARIABLE = .{},
    gen_busy: bool = false,

    fn nextId(self: *ServerState) u64 {
        return self.req_counter.fetchAdd(1, .monotonic);
    }

    fn acquireSlot(self: *ServerState, sock: SOCKET) bool {
        AcquireSRWLockExclusive(&self.gen_lock);
        defer ReleaseSRWLockExclusive(&self.gen_lock);
        while (self.gen_busy) {
            if (isSocketDisconnected(sock)) return false;
            _ = SleepConditionVariableSRW(&self.gen_cv, &self.gen_lock, 50, 0);
        }
        if (isSocketDisconnected(sock)) return false;
        self.gen_busy = true;
        return true;
    }

    fn releaseSlot(self: *ServerState) void {
        AcquireSRWLockExclusive(&self.gen_lock);
        self.gen_busy = false;
        ReleaseSRWLockExclusive(&self.gen_lock);
        WakeConditionVariable(&self.gen_cv);
    }
};

// ============================================================================
// HTTP Request reader (Content-Length up to 32 MB, chunked, Expect: 100-continue, keep-alive)
// ============================================================================

const HttpRequest = struct {
    method: []const u8,
    path: []const u8,
    http_minor: u8,
    origin: ?[]const u8,
    keep_alive: bool,
    body: []const u8,
};

const ConnBuf = struct {
    sock: SOCKET,
    alloc: Allocator,
    buf: std.ArrayList(u8) = .empty,
    pos: usize = 0,

    fn deinit(self: *ConnBuf) void {
        self.buf.deinit(self.alloc);
    }

    fn available(self: *const ConnBuf) []const u8 {
        return self.buf.items[self.pos..];
    }

    fn consume(self: *ConnBuf, n: usize) void {
        self.pos += n;
        if (self.pos >= self.buf.items.len) {
            self.buf.clearRetainingCapacity();
            self.pos = 0;
        } else if (self.pos > 64 * 1024 and self.pos * 2 >= self.buf.items.len) {
            const rem = self.buf.items.len - self.pos;
            std.mem.copyForwards(u8, self.buf.items[0..rem], self.buf.items[self.pos..]);
            self.buf.items.len = rem;
            self.pos = 0;
        }
    }

    fn fillMore(self: *ConnBuf) !bool {
        var tmp: [16384]u8 = undefined;
        const n = recv(self.sock, &tmp, @intCast(tmp.len), 0);
        if (n <= 0) return false;
        try self.buf.appendSlice(self.alloc, tmp[0..@intCast(n)]);
        return true;
    }

    fn readExact(self: *ConnBuf, out_alloc: Allocator, len: usize) ![]u8 {
        const result = try out_alloc.alloc(u8, len);
        var filled: usize = 0;
        if (self.available().len > 0) {
            const take = @min(self.available().len, len);
            @memcpy(result[0..take], self.available()[0..take]);
            self.consume(take);
            filled += take;
        }
        while (filled < len) {
            const want = len - filled;
            const ask: c_int = @intCast(@min(want, @as(usize, 64 * 1024)));
            const n = recv(self.sock, result.ptr + filled, ask, 0);
            if (n <= 0) return error.UnexpectedEof;
            filled += @intCast(n);
        }
        return result;
    }

    fn readLineCrlf(self: *ConnBuf, max_len: usize) ![]u8 {
        while (true) {
            const avail = self.available();
            if (std.mem.indexOf(u8, avail, "\r\n")) |idx| {
                const line = try self.alloc.dupe(u8, avail[0..idx]);
                self.consume(idx + 2);
                return line;
            }
            if (avail.len > max_len) return error.LineTooLong;
            if (!try self.fillMore()) return error.UnexpectedEof;
        }
    }

    fn readChunkedBody(self: *ConnBuf, out_alloc: Allocator) ![]u8 {
        var body: std.ArrayList(u8) = .empty;
        while (true) {
            const line = try self.readLineCrlf(4096);
            defer self.alloc.free(line);
            const semi = std.mem.indexOfScalar(u8, line, ';') orelse line.len;
            const size_str = std.mem.trim(u8, line[0..semi], " \t");
            if (size_str.len == 0) return error.InvalidChunkedEncoding;
            const chunk_size = std.fmt.parseInt(usize, size_str, 16) catch return error.InvalidChunkedEncoding;

            if (chunk_size == 0) {
                // Consume trailer lines until empty CRLF line
                while (true) {
                    const tr = try self.readLineCrlf(4096);
                    defer self.alloc.free(tr);
                    if (tr.len == 0) break;
                }
                return body.toOwnedSlice(out_alloc);
            }

            if (body.items.len + chunk_size > MAX_BODY_BYTES) {
                return error.PayloadTooLarge;
            }

            try body.ensureTotalCapacity(out_alloc, body.items.len + chunk_size);
            var rem = chunk_size;
            while (rem > 0) {
                if (self.available().len == 0) {
                    if (!try self.fillMore()) return error.UnexpectedEof;
                }
                const take = @min(self.available().len, rem);
                body.appendSliceAssumeCapacity(self.available()[0..take]);
                self.consume(take);
                rem -= take;
            }

            // Trailing CRLF after chunk payload
            while (self.available().len < 2) {
                if (!try self.fillMore()) return error.UnexpectedEof;
            }
            if (self.available()[0] != '\r' or self.available()[1] != '\n') {
                return error.InvalidChunkedEncoding;
            }
            self.consume(2);
        }
    }
};

const ReadReqError = error{
    Eof,
    BadRequest,
    PayloadTooLarge,
    OutOfMemory,
};

fn headerContainsToken(value: []const u8, token: []const u8) bool {
    var it = std.mem.splitScalar(u8, value, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t");
        if (std.ascii.eqlIgnoreCase(trimmed, token)) return true;
    }
    return false;
}

fn readHttpRequest(conn: *ConnBuf, req_alloc: Allocator) ReadReqError!HttpRequest {
    // Wait for end of headers "\r\n\r\n"
    var hdr_end: usize = 0;
    while (true) {
        const avail = conn.available();
        if (std.mem.indexOf(u8, avail, "\r\n\r\n")) |idx| {
            hdr_end = idx;
            break;
        }
        if (avail.len > MAX_HEADER_BYTES) return error.BadRequest;
        const ok = conn.fillMore() catch return error.OutOfMemory;
        if (!ok) {
            if (conn.available().len == 0) return error.Eof;
            return error.BadRequest;
        }
    }

    const raw_headers = try req_alloc.dupe(u8, conn.available()[0..hdr_end]);
    conn.consume(hdr_end + 4);

    var lines = std.mem.splitSequence(u8, raw_headers, "\r\n");
    const req_line = lines.next() orelse return error.BadRequest;
    var parts = std.mem.splitScalar(u8, req_line, ' ');
    const method = parts.next() orelse return error.BadRequest;
    const raw_target = parts.next() orelse return error.BadRequest;
    const version = parts.next() orelse return error.BadRequest;
    if (parts.next() != null) return error.BadRequest;
    if (!std.mem.startsWith(u8, version, "HTTP/1.")) return error.BadRequest;
    const minor: u8 = if (version.len >= 8 and version[7] == '0') 0 else 1;

    const qpos = std.mem.indexOfScalar(u8, raw_target, '?') orelse raw_target.len;
    const path = raw_target[0..qpos];

    var content_length: ?usize = null;
    var is_chunked = false;
    var expect_continue = false;
    var origin: ?[]const u8 = null;
    var keep_alive: bool = (minor >= 1);

    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.BadRequest;
        const name = std.mem.trim(u8, line[0..colon], " \t");
        const val = std.mem.trim(u8, line[colon + 1 ..], " \t");

        if (std.ascii.eqlIgnoreCase(name, "content-length")) {
            content_length = std.fmt.parseInt(usize, val, 10) catch return error.BadRequest;
        } else if (std.ascii.eqlIgnoreCase(name, "transfer-encoding")) {
            if (headerContainsToken(val, "chunked")) {
                is_chunked = true;
            }
        } else if (std.ascii.eqlIgnoreCase(name, "expect")) {
            if (std.ascii.eqlIgnoreCase(val, "100-continue")) {
                expect_continue = true;
            }
        } else if (std.ascii.eqlIgnoreCase(name, "origin")) {
            origin = val;
        } else if (std.ascii.eqlIgnoreCase(name, "connection")) {
            if (headerContainsToken(val, "close")) {
                keep_alive = false;
            } else if (headerContainsToken(val, "keep-alive")) {
                keep_alive = true;
            }
        }
    }

    if (content_length) |cl| {
        if (cl > MAX_BODY_BYTES) return error.PayloadTooLarge;
    }

    if (expect_continue) {
        if (!sendAll(conn.sock, "HTTP/1.1 100 Continue\r\n\r\n")) {
            return error.Eof;
        }
    }

    var body: []const u8 = "";
    if (is_chunked) {
        body = conn.readChunkedBody(req_alloc) catch |err| switch (err) {
            error.PayloadTooLarge => return error.PayloadTooLarge,
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.BadRequest,
        };
    } else if (content_length) |cl| {
        if (cl > 0) {
            body = conn.readExact(req_alloc, cl) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.BadRequest,
            };
        }
    }

    return .{
        .method = method,
        .path = path,
        .http_minor = minor,
        .origin = origin,
        .keep_alive = keep_alive,
        .body = body,
    };
}

// ============================================================================
// HTTP Response helpers
// ============================================================================

fn statusText(code: u16) []const u8 {
    return switch (code) {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        else => "Error",
    };
}

fn appendCorsHeaders(alloc: Allocator, hdr: *std.ArrayList(u8), origin: ?[]const u8, is_preflight: bool) Allocator.Error!void {
    if (origin) |orig| {
        if (isAllowedOrigin(orig)) {
            try hdr.print(alloc, "Access-Control-Allow-Origin: {s}\r\nVary: Origin\r\n", .{orig});
            if (is_preflight) {
                try hdr.appendSlice(alloc, "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Max-Age: 600\r\n");
            }
        }
    }
}

fn sendResponse(
    alloc: Allocator,
    sock: SOCKET,
    status: u16,
    content_type: []const u8,
    body: []const u8,
    origin: ?[]const u8,
    keep_alive: bool,
) bool {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(alloc);

    hdr.print(alloc, "HTTP/1.1 {d} {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: {s}\r\n", .{
        status,
        statusText(status),
        content_type,
        body.len,
        if (keep_alive) "keep-alive" else "close",
    }) catch return false;
    appendCorsHeaders(alloc, &hdr, origin, false) catch return false;
    hdr.appendSlice(alloc, "\r\n") catch return false;

    if (!sendAll(sock, hdr.items)) return false;
    if (body.len > 0) {
        if (!sendAll(sock, body)) return false;
    }
    return true;
}

fn sendJsonError(
    alloc: Allocator,
    sock: SOCKET,
    status: u16,
    message: []const u8,
    err_type: []const u8,
    err_code: ?[]const u8,
    origin: ?[]const u8,
    keep_alive: bool,
) bool {
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer };
    jw.beginObject() catch return false;
    jw.objectField("error") catch return false;
    jw.beginObject() catch return false;
    jw.objectField("message") catch return false;
    jw.write(message) catch return false;
    jw.objectField("type") catch return false;
    jw.write(err_type) catch return false;
    jw.objectField("code") catch return false;
    if (err_code) |c| {
        jw.write(c) catch return false;
    } else {
        jw.write(null) catch return false;
    }
    jw.endObject() catch return false;
    jw.endObject() catch return false;

    return sendResponse(alloc, sock, status, "application/json", out.written(), origin, keep_alive);
}

fn sendSseHeaders(alloc: Allocator, sock: SOCKET, origin: ?[]const u8, keep_alive: bool) bool {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(alloc);

    hdr.print(alloc, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: {s}\r\n", .{
        if (keep_alive) "keep-alive" else "close",
    }) catch return false;
    appendCorsHeaders(alloc, &hdr, origin, false) catch return false;
    hdr.appendSlice(alloc, "\r\n") catch return false;

    return sendAll(sock, hdr.items);
}

// ============================================================================
// JSON parameter parsing & usage/timings serialization
// ============================================================================

fn jsonAsF32(val: std.json.Value) ?f32 {
    return switch (val) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f32, s) catch null,
        else => null,
    };
}

fn jsonAsU32(val: std.json.Value) ?u32 {
    return switch (val) {
        .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
        .float => |f| if (f >= 0 and f <= @as(f64, @floatFromInt(std.math.maxInt(u32)))) @intFromFloat(f) else null,
        else => null,
    };
}

fn jsonAsU64(val: std.json.Value) ?u64 {
    return switch (val) {
        .integer => |i| if (i >= 0) @intCast(i) else @bitCast(i),
        .number_string => |s| std.fmt.parseInt(u64, s, 10) catch null,
        else => null,
    };
}

const ParsedCommonParams = struct {
    gen: GenParams,
    stream: bool,
    include_usage: bool,
};

fn parseCommonParams(alloc: Allocator, obj: std.json.ObjectMap) !ParsedCommonParams {
    var max_tokens: u32 = 2048;
    if (obj.get("max_completion_tokens")) |v| {
        if (v != .null) max_tokens = jsonAsU32(v) orelse return error.InvalidParam;
    } else if (obj.get("max_tokens")) |v| {
        if (v != .null) max_tokens = jsonAsU32(v) orelse return error.InvalidParam;
    }

    var temperature: f32 = 1.0;
    if (obj.get("temperature")) |v| {
        if (v != .null) temperature = jsonAsF32(v) orelse return error.InvalidParam;
    }

    var top_p: f32 = 1.0;
    if (obj.get("top_p")) |v| {
        if (v != .null) top_p = jsonAsF32(v) orelse return error.InvalidParam;
    }

    var top_k: u32 = 0; // model defaults (GGUF general.sampling.*): temp 1, top_p 1, top_k 0, min_p 0
    if (obj.get("top_k")) |v| {
        if (v != .null) top_k = jsonAsU32(v) orelse return error.InvalidParam;
    }

    var min_p: f32 = 0.0;
    if (obj.get("min_p")) |v| {
        if (v != .null) min_p = jsonAsF32(v) orelse return error.InvalidParam;
    }

    var seed: u64 = 0;
    if (obj.get("seed")) |v| {
        if (v != .null) seed = jsonAsU64(v) orelse return error.InvalidParam;
    }

    var cache_prompt: bool = true;
    if (obj.get("cache_prompt")) |v| {
        if (v != .null) {
            if (v != .bool) return error.InvalidParam;
            cache_prompt = v.bool;
        }
    }

    var stream: bool = false;
    if (obj.get("stream")) |v| {
        if (v != .null) {
            if (v != .bool) return error.InvalidParam;
            stream = v.bool;
        }
    }

    var include_usage: bool = false;
    if (obj.get("stream_options")) |so| {
        if (so != .null) {
            if (so != .object) return error.InvalidParam;
            if (so.object.get("include_usage")) |iu| {
                if (iu != .null) {
                    if (iu != .bool) return error.InvalidParam;
                    include_usage = iu.bool;
                }
            }
        }
    }

    var stops: std.ArrayList([]const u8) = .empty;
    if (obj.get("stop")) |sv| {
        switch (sv) {
            .null => {},
            .string => |s| {
                if (s.len > 0) try stops.append(alloc, s);
            },
            .array => |arr| {
                for (arr.items) |item| {
                    if (item != .string) return error.InvalidParam;
                    if (item.string.len > 0) try stops.append(alloc, item.string);
                }
            },
            else => return error.InvalidParam,
        }
    }

    return .{
        .gen = .{
            .max_tokens = max_tokens,
            .temperature = temperature,
            .top_p = top_p,
            .top_k = top_k,
            .min_p = min_p,
            .seed = seed,
            .stop = try stops.toOwnedSlice(alloc),
            .cache_prompt = cache_prompt,
        },
        .stream = stream,
        .include_usage = include_usage,
    };
}

const ResolvedStats = struct {
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
    prompt_n: u32,
    prompt_ms: f64,
    prompt_per_second: f64,
    predicted_n: u32,
    predicted_ms: f64,
    predicted_per_second: f64,
    cache_n: u32,
};

fn computeResolvedStats(stats: GenStats, prompt_len: u32, generated_tokens: u32, wall_ms: f64) ResolvedStats {
    const total_prompt = if (stats.prompt_tokens > 0) stats.prompt_tokens else prompt_len;
    const cached = @min(stats.cached_tokens, total_prompt);
    const prompt_n = total_prompt - cached;
    const comp = if (stats.completion_tokens > 0) stats.completion_tokens else generated_tokens;
    const p_ms = stats.prompt_ms;
    const d_ms = if (stats.decode_ms > 0.0) stats.decode_ms else wall_ms;
    const p_tps = if (p_ms > 0.0) @as(f64, @floatFromInt(prompt_n)) * 1000.0 / p_ms else 0.0;
    const d_tps = if (d_ms > 0.0) @as(f64, @floatFromInt(comp)) * 1000.0 / d_ms else 0.0;

    return .{
        .prompt_tokens = total_prompt,
        .cached_tokens = cached,
        .completion_tokens = comp,
        .prompt_n = prompt_n,
        .prompt_ms = p_ms,
        .prompt_per_second = p_tps,
        .predicted_n = comp,
        .predicted_ms = d_ms,
        .predicted_per_second = d_tps,
        .cache_n = cached,
    };
}

fn writeUsageField(jw: *std.json.Stringify, rs: ResolvedStats) !void {
    try jw.objectField("usage");
    try jw.beginObject();
    try jw.objectField("prompt_tokens");
    try jw.write(rs.prompt_tokens);
    try jw.objectField("completion_tokens");
    try jw.write(rs.completion_tokens);
    try jw.objectField("total_tokens");
    try jw.write(rs.prompt_tokens + rs.completion_tokens);
    try jw.objectField("prompt_tokens_details");
    try jw.beginObject();
    try jw.objectField("cached_tokens");
    try jw.write(rs.cached_tokens);
    try jw.endObject();
    try jw.endObject();
}

fn writeTimingsField(jw: *std.json.Stringify, rs: ResolvedStats) !void {
    try jw.objectField("timings");
    try jw.beginObject();
    try jw.objectField("prompt_n");
    try jw.write(rs.prompt_n);
    try jw.objectField("prompt_ms");
    try jw.write(rs.prompt_ms);
    try jw.objectField("prompt_per_second");
    try jw.write(rs.prompt_per_second);
    try jw.objectField("predicted_n");
    try jw.write(rs.predicted_n);
    try jw.objectField("predicted_ms");
    try jw.write(rs.predicted_ms);
    try jw.objectField("predicted_per_second");
    try jw.write(rs.predicted_per_second);
    try jw.objectField("cache_n");
    try jw.write(rs.cache_n);
    try jw.endObject();
}

fn lookupSpecials(engine: *Engine) chat_template.SpecialIds {
    return .{
        .message_model = engine.special(engine.ctx, chat_template.TOK_MSG_MODEL),
        .content_thinking = engine.special(engine.ctx, chat_template.TOK_CONTENT_THINKING),
        .content_text = engine.special(engine.ctx, chat_template.TOK_CONTENT_TEXT),
        .content_tool_json = engine.special(engine.ctx, chat_template.TOK_CONTENT_TOOL_JSON),
        .end_message = engine.special(engine.ctx, chat_template.TOK_END_MESSAGE),
        .end_sampling = engine.special(engine.ctx, chat_template.TOK_END_SAMPLING),
        .message_user = engine.special(engine.ctx, chat_template.TOK_MSG_USER),
        .message_system = engine.special(engine.ctx, chat_template.TOK_MSG_SYSTEM),
        .message_tool = engine.special(engine.ctx, chat_template.TOK_MSG_TOOL),
        .content_xml = engine.special(engine.ctx, chat_template.TOK_CONTENT_XML),
        .content_image = engine.special(engine.ctx, chat_template.TOK_CONTENT_IMAGE),
    };
}

// ============================================================================
// Endpoint handlers
// ============================================================================

fn handleTokenize(state: *ServerState, alloc: Allocator, sock: SOCKET, req: HttpRequest) bool {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, alloc, req.body, .{}) catch {
        return sendJsonError(alloc, sock, 400, "invalid JSON body", "invalid_request_error", null, req.origin, req.keep_alive);
    };
    if (parsed != .object) {
        return sendJsonError(alloc, sock, 400, "JSON body must be an object", "invalid_request_error", null, req.origin, req.keep_alive);
    }

    const text_val = parsed.object.get("content") orelse parsed.object.get("text") orelse {
        return sendJsonError(alloc, sock, 400, "missing 'content' or 'text' string field", "invalid_request_error", null, req.origin, req.keep_alive);
    };
    if (text_val != .string) {
        return sendJsonError(alloc, sock, 400, "'content' must be a string", "invalid_request_error", null, req.origin, req.keep_alive);
    }

    var parse_special = false;
    if (parsed.object.get("parse_special") orelse parsed.object.get("add_special")) |ps| {
        if (ps == .bool) parse_special = ps.bool;
    }

    const tokens = state.engine.tokenize(state.engine.ctx, alloc, text_val.string, parse_special) catch {
        return sendJsonError(alloc, sock, 500, "tokenization failed", "server_error", null, req.origin, req.keep_alive);
    };

    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer };
    jw.beginObject() catch return false;
    jw.objectField("tokens") catch return false;
    jw.write(tokens) catch return false;
    jw.endObject() catch return false;

    return sendResponse(alloc, sock, 200, "application/json", out.written(), req.origin, req.keep_alive);
}

// ---- /v1/chat/completions ----

const ChatGenCtx = struct {
    alloc: Allocator,
    sock: SOCKET,
    engine: *Engine,
    parser: chat_template.OutputParser,
    stream: bool,
    id: []const u8,
    created: i64,
    model: []const u8,
    client_disconnected: bool = false,
    stopped_by_stop_string: bool = false,
    stopped_by_eog: bool = false,
    generated_tokens: u32 = 0,
    // Non-stream accumulators
    acc_reasoning: std.ArrayList(u8) = .empty,
    acc_content: std.ArrayList(u8) = .empty,
    acc_tools: std.ArrayList(chat_template.ParsedToolCall) = .empty,
    has_tool_calls: bool = false,

    fn sendSseDelta(self: *ChatGenCtx, event: chat_template.OutputEvent) bool {
        var out: std.Io.Writer.Allocating = .init(self.alloc);
        defer out.deinit();
        out.writer.writeAll("data: ") catch return false;

        var jw: std.json.Stringify = .{ .writer = &out.writer };
        jw.beginObject() catch return false;
        jw.objectField("id") catch return false;
        jw.write(self.id) catch return false;
        jw.objectField("object") catch return false;
        jw.write("chat.completion.chunk") catch return false;
        jw.objectField("created") catch return false;
        jw.write(self.created) catch return false;
        jw.objectField("model") catch return false;
        jw.write(self.model) catch return false;
        jw.objectField("choices") catch return false;
        jw.beginArray() catch return false;
        jw.beginObject() catch return false;
        jw.objectField("index") catch return false;
        jw.write(@as(u32, 0)) catch return false;
        jw.objectField("delta") catch return false;
        jw.beginObject() catch return false;

        switch (event) {
            .reasoning => |s| {
                jw.objectField("reasoning_content") catch return false;
                jw.write(s) catch return false;
            },
            .content => |s| {
                jw.objectField("content") catch return false;
                jw.write(s) catch return false;
            },
            .tool_call => |tc| {
                jw.objectField("tool_calls") catch return false;
                jw.beginArray() catch return false;
                jw.beginObject() catch return false;
                jw.objectField("index") catch return false;
                jw.write(tc.index) catch return false;
                jw.objectField("id") catch return false;
                jw.write(tc.id) catch return false;
                jw.objectField("type") catch return false;
                jw.write("function") catch return false;
                jw.objectField("function") catch return false;
                jw.beginObject() catch return false;
                jw.objectField("name") catch return false;
                jw.write(tc.name) catch return false;
                jw.objectField("arguments") catch return false;
                jw.write(tc.arguments) catch return false;
                jw.endObject() catch return false;
                jw.endObject() catch return false;
                jw.endArray() catch return false;
            },
            else => unreachable,
        }

        jw.endObject() catch return false;
        jw.objectField("finish_reason") catch return false;
        jw.write(null) catch return false;
        jw.endObject() catch return false;
        jw.endArray() catch return false;
        jw.endObject() catch return false;

        out.writer.writeAll("\n\n") catch return false;
        return sendChunk(self.sock, out.written());
    }

    fn processEvents(self: *ChatGenCtx, events: []const chat_template.OutputEvent) bool {
        for (events) |ev| {
            switch (ev) {
                .reasoning => |s| {
                    if (self.stream) {
                        if (!self.sendSseDelta(ev)) {
                            self.client_disconnected = true;
                            return false;
                        }
                    } else {
                        self.acc_reasoning.appendSlice(self.alloc, s) catch return false;
                    }
                },
                .content => |s| {
                    if (self.stream) {
                        if (!self.sendSseDelta(ev)) {
                            self.client_disconnected = true;
                            return false;
                        }
                    } else {
                        self.acc_content.appendSlice(self.alloc, s) catch return false;
                    }
                },
                .tool_call => |tc| {
                    self.has_tool_calls = true;
                    if (self.stream) {
                        if (!self.sendSseDelta(ev)) {
                            self.client_disconnected = true;
                            return false;
                        }
                    } else {
                        self.acc_tools.append(self.alloc, tc) catch return false;
                    }
                },
                .stopped_by_stop_string => {
                    self.stopped_by_stop_string = true;
                    return false;
                },
                .stopped_by_eog => {
                    self.stopped_by_eog = true;
                    return false;
                },
            }
        }
        return true;
    }

    fn onToken(user: *anyopaque, id: u32, bytes: []const u8) bool {
        const self: *ChatGenCtx = @ptrCast(@alignCast(user));
        if (isSocketDisconnected(self.sock)) {
            self.client_disconnected = true;
            return false;
        }
        self.generated_tokens += 1;

        var events: std.ArrayList(chat_template.OutputEvent) = .empty;
        const is_eog = self.engine.is_eog(self.engine.ctx, id);
        self.parser.feed(id, bytes, is_eog, &events) catch {
            self.client_disconnected = true;
            return false;
        };
        return self.processEvents(events.items);
    }
};

fn sendChatRoleChunk(alloc: Allocator, sock: SOCKET, id: []const u8, created: i64, model: []const u8) bool {
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    out.writer.writeAll("data: ") catch return false;
    var jw: std.json.Stringify = .{ .writer = &out.writer };
    jw.beginObject() catch return false;
    jw.objectField("id") catch return false;
    jw.write(id) catch return false;
    jw.objectField("object") catch return false;
    jw.write("chat.completion.chunk") catch return false;
    jw.objectField("created") catch return false;
    jw.write(created) catch return false;
    jw.objectField("model") catch return false;
    jw.write(model) catch return false;
    jw.objectField("choices") catch return false;
    jw.beginArray() catch return false;
    jw.beginObject() catch return false;
    jw.objectField("index") catch return false;
    jw.write(@as(u32, 0)) catch return false;
    jw.objectField("delta") catch return false;
    jw.beginObject() catch return false;
    jw.objectField("role") catch return false;
    jw.write("assistant") catch return false;
    jw.endObject() catch return false;
    jw.objectField("finish_reason") catch return false;
    jw.write(null) catch return false;
    jw.endObject() catch return false;
    jw.endArray() catch return false;
    jw.endObject() catch return false;
    out.writer.writeAll("\n\n") catch return false;
    return sendChunk(sock, out.written());
}

fn handleChatCompletions(state: *ServerState, alloc: Allocator, sock: SOCKET, req: HttpRequest) bool {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, alloc, req.body, .{}) catch {
        return sendJsonError(alloc, sock, 400, "invalid JSON body", "invalid_request_error", null, req.origin, req.keep_alive);
    };
    if (parsed != .object) {
        return sendJsonError(alloc, sock, 400, "JSON body must be an object", "invalid_request_error", null, req.origin, req.keep_alive);
    }

    const messages_val = parsed.object.get("messages") orelse {
        return sendJsonError(alloc, sock, 400, "missing 'messages' field", "invalid_request_error", null, req.origin, req.keep_alive);
    };

    const params = parseCommonParams(alloc, parsed.object) catch {
        return sendJsonError(alloc, sock, 400, "invalid generation parameter", "invalid_request_error", null, req.origin, req.keep_alive);
    };

    var tpl_err: []const u8 = "chat template error";
    const segments = chat_template.render(
        alloc,
        messages_val,
        parsed.object.get("tools"),
        parsed.object.get("reasoning_effort"),
        &tpl_err,
    ) catch |err| {
        if (err == error.OutOfMemory) {
            return sendJsonError(alloc, sock, 500, "out of memory", "server_error", null, req.origin, req.keep_alive);
        }
        return sendJsonError(alloc, sock, 400, tpl_err, "invalid_request_error", null, req.origin, req.keep_alive);
    };

    // Tokenize each segment separately so user-supplied text can NEVER inject special tokens
    var prompt_tokens: std.ArrayList(u32) = .empty;
    for (segments) |seg| {
        const toks = state.engine.tokenize(state.engine.ctx, alloc, seg.text, seg.parse_special) catch {
            return sendJsonError(alloc, sock, 500, "tokenization failed", "server_error", null, req.origin, req.keep_alive);
        };
        prompt_tokens.appendSlice(alloc, toks) catch {
            return sendJsonError(alloc, sock, 500, "out of memory", "server_error", null, req.origin, req.keep_alive);
        };
    }

    if (prompt_tokens.items.len > state.engine.n_ctx) {
        const msg = std.fmt.allocPrint(alloc, "prompt length ({d}) exceeds n_ctx ({d})", .{
            prompt_tokens.items.len,
            state.engine.n_ctx,
        }) catch "prompt exceeds n_ctx";
        return sendJsonError(alloc, sock, 400, msg, "invalid_request_error", "context_length_exceeded", req.origin, req.keep_alive);
    }

    // Acquire single-slot generation mutex (checking disconnect while waiting)
    if (!state.acquireSlot(sock)) {
        return false; // Client disconnected while queued
    }
    defer state.releaseSlot();

    const created = unixTimestampSeconds();
    const id = std.fmt.allocPrint(alloc, "chatcmpl-{d}", .{state.nextId()}) catch "chatcmpl-0";

    if (params.stream) {
        if (!sendSseHeaders(alloc, sock, req.origin, req.keep_alive)) return false;
        if (!sendChatRoleChunk(alloc, sock, id, created, state.model_name)) return false;
    }

    var gen_ctx: ChatGenCtx = .{
        .alloc = alloc,
        .sock = sock,
        .engine = state.engine,
        .parser = chat_template.OutputParser.init(alloc, lookupSpecials(state.engine), params.gen.stop),
        .stream = params.stream,
        .id = id,
        .created = created,
        .model = state.model_name,
    };
    defer gen_ctx.parser.deinit();

    var stats: GenStats = .{ .prompt_tokens = @intCast(prompt_tokens.items.len) };
    var t_start: i64 = 0;
    var t_end: i64 = 0;
    _ = QueryPerformanceCounter(&t_start);

    const raw_reason = state.engine.generate(
        state.engine.ctx,
        prompt_tokens.items,
        params.gen,
        &stats,
        @ptrCast(&gen_ctx),
        ChatGenCtx.onToken,
    ) catch {
        if (params.stream) {
            _ = sendChunkEnd(sock);
            return false;
        }
        return sendJsonError(alloc, sock, 500, "generation failed", "server_error", null, req.origin, req.keep_alive);
    };

    _ = QueryPerformanceCounter(&t_end);

    if (gen_ctx.client_disconnected or (raw_reason == .aborted and !gen_ctx.stopped_by_stop_string and !gen_ctx.stopped_by_eog)) {
        return false;
    }

    // Flush any buffered UTF-8 / stop hold-back / pending tool call
    var tail_events: std.ArrayList(chat_template.OutputEvent) = .empty;
    gen_ctx.parser.finish(&tail_events) catch return false;
    if (!gen_ctx.processEvents(tail_events.items)) {
        if (gen_ctx.client_disconnected) return false;
    }

    const finish_str: []const u8 = if (raw_reason == .length)
        "length"
    else if (gen_ctx.has_tool_calls)
        "tool_calls"
    else
        "stop";

    const wall_ms = perfCounterMs(t_start, t_end, state.qpc_freq);
    const rs = computeResolvedStats(stats, @intCast(prompt_tokens.items.len), gen_ctx.generated_tokens, wall_ms);

    if (params.stream) {
        // Finish chunk
        {
            var out: std.Io.Writer.Allocating = .init(alloc);
            defer out.deinit();
            out.writer.writeAll("data: ") catch return false;
            var jw: std.json.Stringify = .{ .writer = &out.writer };
            jw.beginObject() catch return false;
            jw.objectField("id") catch return false;
            jw.write(id) catch return false;
            jw.objectField("object") catch return false;
            jw.write("chat.completion.chunk") catch return false;
            jw.objectField("created") catch return false;
            jw.write(created) catch return false;
            jw.objectField("model") catch return false;
            jw.write(state.model_name) catch return false;
            jw.objectField("choices") catch return false;
            jw.beginArray() catch return false;
            jw.beginObject() catch return false;
            jw.objectField("index") catch return false;
            jw.write(@as(u32, 0)) catch return false;
            jw.objectField("delta") catch return false;
            jw.beginObject() catch return false;
            jw.endObject() catch return false;
            jw.objectField("finish_reason") catch return false;
            jw.write(finish_str) catch return false;
            jw.endObject() catch return false;
            jw.endArray() catch return false;
            if (!params.include_usage) {
                writeTimingsField(&jw, rs) catch return false;
            }
            jw.endObject() catch return false;
            out.writer.writeAll("\n\n") catch return false;
            if (!sendChunk(sock, out.written())) return false;
        }

        // Optional final usage + timings chunk when stream_options.include_usage is true
        if (params.include_usage) {
            var out: std.Io.Writer.Allocating = .init(alloc);
            defer out.deinit();
            out.writer.writeAll("data: ") catch return false;
            var jw: std.json.Stringify = .{ .writer = &out.writer };
            jw.beginObject() catch return false;
            jw.objectField("id") catch return false;
            jw.write(id) catch return false;
            jw.objectField("object") catch return false;
            jw.write("chat.completion.chunk") catch return false;
            jw.objectField("created") catch return false;
            jw.write(created) catch return false;
            jw.objectField("model") catch return false;
            jw.write(state.model_name) catch return false;
            jw.objectField("choices") catch return false;
            jw.beginArray() catch return false;
            jw.endArray() catch return false;
            writeUsageField(&jw, rs) catch return false;
            writeTimingsField(&jw, rs) catch return false;
            jw.endObject() catch return false;
            out.writer.writeAll("\n\n") catch return false;
            if (!sendChunk(sock, out.written())) return false;
        }

        if (!sendChunk(sock, "data: [DONE]\n\n")) return false;
        return sendChunkEnd(sock);
    }

    // Non-streaming JSON response
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer };
    jw.beginObject() catch return false;
    jw.objectField("id") catch return false;
    jw.write(id) catch return false;
    jw.objectField("object") catch return false;
    jw.write("chat.completion") catch return false;
    jw.objectField("created") catch return false;
    jw.write(created) catch return false;
    jw.objectField("model") catch return false;
    jw.write(state.model_name) catch return false;
    jw.objectField("choices") catch return false;
    jw.beginArray() catch return false;
    jw.beginObject() catch return false;
    jw.objectField("index") catch return false;
    jw.write(@as(u32, 0)) catch return false;
    jw.objectField("message") catch return false;
    jw.beginObject() catch return false;
    jw.objectField("role") catch return false;
    jw.write("assistant") catch return false;
    jw.objectField("content") catch return false;
    if (gen_ctx.acc_content.items.len == 0 and gen_ctx.acc_tools.items.len > 0) {
        jw.write(null) catch return false;
    } else {
        jw.write(gen_ctx.acc_content.items) catch return false;
    }
    if (gen_ctx.acc_reasoning.items.len > 0) {
        jw.objectField("reasoning_content") catch return false;
        jw.write(gen_ctx.acc_reasoning.items) catch return false;
    }
    if (gen_ctx.acc_tools.items.len > 0) {
        jw.objectField("tool_calls") catch return false;
        jw.beginArray() catch return false;
        for (gen_ctx.acc_tools.items) |tc| {
            jw.beginObject() catch return false;
            jw.objectField("id") catch return false;
            jw.write(tc.id) catch return false;
            jw.objectField("type") catch return false;
            jw.write("function") catch return false;
            jw.objectField("function") catch return false;
            jw.beginObject() catch return false;
            jw.objectField("name") catch return false;
            jw.write(tc.name) catch return false;
            jw.objectField("arguments") catch return false;
            jw.write(tc.arguments) catch return false;
            jw.endObject() catch return false;
            jw.endObject() catch return false;
        }
        jw.endArray() catch return false;
    }
    jw.endObject() catch return false;
    jw.objectField("finish_reason") catch return false;
    jw.write(finish_str) catch return false;
    jw.endObject() catch return false;
    jw.endArray() catch return false;
    writeUsageField(&jw, rs) catch return false;
    writeTimingsField(&jw, rs) catch return false;
    jw.endObject() catch return false;

    return sendResponse(alloc, sock, 200, "application/json", out.written(), req.origin, req.keep_alive);
}

// ---- /v1/completions ----

const CmplGenCtx = struct {
    alloc: Allocator,
    sock: SOCKET,
    engine: *Engine,
    stops: []const []const u8,
    stream: bool,
    id: []const u8,
    created: i64,
    model: []const u8,
    client_disconnected: bool = false,
    stopped_by_stop_string: bool = false,
    stopped_by_eog: bool = false,
    generated_tokens: u32 = 0,
    hold_buf: std.ArrayList(u8) = .empty,
    acc_text: std.ArrayList(u8) = .empty,

    fn emitText(self: *CmplGenCtx, s: []const u8) bool {
        if (s.len == 0) return true;
        if (!self.stream) {
            self.acc_text.appendSlice(self.alloc, s) catch return false;
            return true;
        }
        var out: std.Io.Writer.Allocating = .init(self.alloc);
        defer out.deinit();
        out.writer.writeAll("data: ") catch return false;
        var jw: std.json.Stringify = .{ .writer = &out.writer };
        jw.beginObject() catch return false;
        jw.objectField("id") catch return false;
        jw.write(self.id) catch return false;
        jw.objectField("object") catch return false;
        jw.write("text_completion") catch return false;
        jw.objectField("created") catch return false;
        jw.write(self.created) catch return false;
        jw.objectField("model") catch return false;
        jw.write(self.model) catch return false;
        jw.objectField("choices") catch return false;
        jw.beginArray() catch return false;
        jw.beginObject() catch return false;
        jw.objectField("index") catch return false;
        jw.write(@as(u32, 0)) catch return false;
        jw.objectField("text") catch return false;
        jw.write(s) catch return false;
        jw.objectField("logprobs") catch return false;
        jw.write(null) catch return false;
        jw.objectField("finish_reason") catch return false;
        jw.write(null) catch return false;
        jw.endObject() catch return false;
        jw.endArray() catch return false;
        jw.endObject() catch return false;
        out.writer.writeAll("\n\n") catch return false;
        if (!sendChunk(self.sock, out.written())) {
            self.client_disconnected = true;
            return false;
        }
        return true;
    }

    fn flush(self: *CmplGenCtx, force_all: bool) bool {
        if (self.hold_buf.items.len == 0) return true;
        if (self.stops.len > 0) {
            if (chat_template.findEarliestStop(self.hold_buf.items, self.stops)) |pos| {
                if (pos > 0) {
                    if (!self.emitText(self.hold_buf.items[0..pos])) return false;
                }
                self.hold_buf.clearRetainingCapacity();
                self.stopped_by_stop_string = true;
                return false;
            }
        }
        var hold: usize = 0;
        if (!force_all) {
            const stop_hold = chat_template.longestStopPrefixAtEnd(self.hold_buf.items, self.stops);
            const prefix_len = self.hold_buf.items.len - stop_hold;
            const utf8_hold = chat_template.incompleteUtf8TailLen(self.hold_buf.items[0..prefix_len]);
            hold = stop_hold + utf8_hold;
        }
        const emit_len = self.hold_buf.items.len - hold;
        if (emit_len > 0) {
            if (!self.emitText(self.hold_buf.items[0..emit_len])) return false;
            if (hold > 0) {
                std.mem.copyForwards(u8, self.hold_buf.items[0..hold], self.hold_buf.items[emit_len..]);
            }
            self.hold_buf.items.len = hold;
        }
        return true;
    }

    fn onToken(user: *anyopaque, id: u32, bytes: []const u8) bool {
        const self: *CmplGenCtx = @ptrCast(@alignCast(user));
        if (isSocketDisconnected(self.sock)) {
            self.client_disconnected = true;
            return false;
        }
        self.generated_tokens += 1;
        if (self.engine.is_eog(self.engine.ctx, id)) {
            self.stopped_by_eog = true;
            return false;
        }
        if (bytes.len == 0) return true;
        self.hold_buf.appendSlice(self.alloc, bytes) catch return false;
        return self.flush(false);
    }
};

fn handleCompletions(state: *ServerState, alloc: Allocator, sock: SOCKET, req: HttpRequest) bool {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, alloc, req.body, .{}) catch {
        return sendJsonError(alloc, sock, 400, "invalid JSON body", "invalid_request_error", null, req.origin, req.keep_alive);
    };
    if (parsed != .object) {
        return sendJsonError(alloc, sock, 400, "JSON body must be an object", "invalid_request_error", null, req.origin, req.keep_alive);
    }

    const prompt_val = parsed.object.get("prompt") orelse {
        return sendJsonError(alloc, sock, 400, "missing 'prompt' field", "invalid_request_error", null, req.origin, req.keep_alive);
    };
    const prompt_str: []const u8 = switch (prompt_val) {
        .string => |s| s,
        .array => |arr| blk: {
            if (arr.items.len == 1 and arr.items[0] == .string) break :blk arr.items[0].string;
            return sendJsonError(alloc, sock, 400, "'prompt' must be a string", "invalid_request_error", null, req.origin, req.keep_alive);
        },
        else => return sendJsonError(alloc, sock, 400, "'prompt' must be a string", "invalid_request_error", null, req.origin, req.keep_alive),
    };

    const params = parseCommonParams(alloc, parsed.object) catch {
        return sendJsonError(alloc, sock, 400, "invalid generation parameter", "invalid_request_error", null, req.origin, req.keep_alive);
    };

    const prompt_tokens = state.engine.tokenize(state.engine.ctx, alloc, prompt_str, false) catch {
        return sendJsonError(alloc, sock, 500, "tokenization failed", "server_error", null, req.origin, req.keep_alive);
    };

    if (prompt_tokens.len > state.engine.n_ctx) {
        const msg = std.fmt.allocPrint(alloc, "prompt length ({d}) exceeds n_ctx ({d})", .{
            prompt_tokens.len,
            state.engine.n_ctx,
        }) catch "prompt exceeds n_ctx";
        return sendJsonError(alloc, sock, 400, msg, "invalid_request_error", "context_length_exceeded", req.origin, req.keep_alive);
    }

    if (!state.acquireSlot(sock)) return false;
    defer state.releaseSlot();

    const created = unixTimestampSeconds();
    const id = std.fmt.allocPrint(alloc, "cmpl-{d}", .{state.nextId()}) catch "cmpl-0";

    if (params.stream) {
        if (!sendSseHeaders(alloc, sock, req.origin, req.keep_alive)) return false;
    }

    var gen_ctx: CmplGenCtx = .{
        .alloc = alloc,
        .sock = sock,
        .engine = state.engine,
        .stops = params.gen.stop,
        .stream = params.stream,
        .id = id,
        .created = created,
        .model = state.model_name,
    };

    var stats: GenStats = .{ .prompt_tokens = @intCast(prompt_tokens.len) };
    var t_start: i64 = 0;
    var t_end: i64 = 0;
    _ = QueryPerformanceCounter(&t_start);

    const raw_reason = state.engine.generate(
        state.engine.ctx,
        prompt_tokens,
        params.gen,
        &stats,
        @ptrCast(&gen_ctx),
        CmplGenCtx.onToken,
    ) catch {
        if (params.stream) {
            _ = sendChunkEnd(sock);
            return false;
        }
        return sendJsonError(alloc, sock, 500, "generation failed", "server_error", null, req.origin, req.keep_alive);
    };

    _ = QueryPerformanceCounter(&t_end);

    if (gen_ctx.client_disconnected or (raw_reason == .aborted and !gen_ctx.stopped_by_stop_string and !gen_ctx.stopped_by_eog)) {
        return false;
    }

    if (!gen_ctx.flush(true) and gen_ctx.client_disconnected) return false;

    const finish_str: []const u8 = if (raw_reason == .length) "length" else "stop";
    const wall_ms = perfCounterMs(t_start, t_end, state.qpc_freq);
    const rs = computeResolvedStats(stats, @intCast(prompt_tokens.len), gen_ctx.generated_tokens, wall_ms);

    if (params.stream) {
        {
            var out: std.Io.Writer.Allocating = .init(alloc);
            defer out.deinit();
            out.writer.writeAll("data: ") catch return false;
            var jw: std.json.Stringify = .{ .writer = &out.writer };
            jw.beginObject() catch return false;
            jw.objectField("id") catch return false;
            jw.write(id) catch return false;
            jw.objectField("object") catch return false;
            jw.write("text_completion") catch return false;
            jw.objectField("created") catch return false;
            jw.write(created) catch return false;
            jw.objectField("model") catch return false;
            jw.write(state.model_name) catch return false;
            jw.objectField("choices") catch return false;
            jw.beginArray() catch return false;
            jw.beginObject() catch return false;
            jw.objectField("index") catch return false;
            jw.write(@as(u32, 0)) catch return false;
            jw.objectField("text") catch return false;
            jw.write("") catch return false;
            jw.objectField("logprobs") catch return false;
            jw.write(null) catch return false;
            jw.objectField("finish_reason") catch return false;
            jw.write(finish_str) catch return false;
            jw.endObject() catch return false;
            jw.endArray() catch return false;
            if (!params.include_usage) {
                writeTimingsField(&jw, rs) catch return false;
            }
            jw.endObject() catch return false;
            out.writer.writeAll("\n\n") catch return false;
            if (!sendChunk(sock, out.written())) return false;
        }
        if (params.include_usage) {
            var out: std.Io.Writer.Allocating = .init(alloc);
            defer out.deinit();
            out.writer.writeAll("data: ") catch return false;
            var jw: std.json.Stringify = .{ .writer = &out.writer };
            jw.beginObject() catch return false;
            jw.objectField("id") catch return false;
            jw.write(id) catch return false;
            jw.objectField("object") catch return false;
            jw.write("text_completion") catch return false;
            jw.objectField("created") catch return false;
            jw.write(created) catch return false;
            jw.objectField("model") catch return false;
            jw.write(state.model_name) catch return false;
            jw.objectField("choices") catch return false;
            jw.beginArray() catch return false;
            jw.endArray() catch return false;
            writeUsageField(&jw, rs) catch return false;
            writeTimingsField(&jw, rs) catch return false;
            jw.endObject() catch return false;
            out.writer.writeAll("\n\n") catch return false;
            if (!sendChunk(sock, out.written())) return false;
        }
        if (!sendChunk(sock, "data: [DONE]\n\n")) return false;
        return sendChunkEnd(sock);
    }

    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer };
    jw.beginObject() catch return false;
    jw.objectField("id") catch return false;
    jw.write(id) catch return false;
    jw.objectField("object") catch return false;
    jw.write("text_completion") catch return false;
    jw.objectField("created") catch return false;
    jw.write(created) catch return false;
    jw.objectField("model") catch return false;
    jw.write(state.model_name) catch return false;
    jw.objectField("choices") catch return false;
    jw.beginArray() catch return false;
    jw.beginObject() catch return false;
    jw.objectField("index") catch return false;
    jw.write(@as(u32, 0)) catch return false;
    jw.objectField("text") catch return false;
    jw.write(gen_ctx.acc_text.items) catch return false;
    jw.objectField("logprobs") catch return false;
    jw.write(null) catch return false;
    jw.objectField("finish_reason") catch return false;
    jw.write(finish_str) catch return false;
    jw.endObject() catch return false;
    jw.endArray() catch return false;
    writeUsageField(&jw, rs) catch return false;
    writeTimingsField(&jw, rs) catch return false;
    jw.endObject() catch return false;

    return sendResponse(alloc, sock, 200, "application/json", out.written(), req.origin, req.keep_alive);
}

fn handleModels(state: *ServerState, alloc: Allocator, sock: SOCKET, req: HttpRequest) bool {
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    var jw: std.json.Stringify = .{ .writer = &out.writer };
    jw.beginObject() catch return false;
    jw.objectField("object") catch return false;
    jw.write("list") catch return false;
    jw.objectField("data") catch return false;
    jw.beginArray() catch return false;
    jw.beginObject() catch return false;
    jw.objectField("id") catch return false;
    jw.write(state.model_name) catch return false;
    jw.objectField("object") catch return false;
    jw.write("model") catch return false;
    jw.objectField("created") catch return false;
    jw.write(state.created_ts) catch return false;
    jw.objectField("owned_by") catch return false;
    jw.write("local") catch return false;
    jw.endObject() catch return false;
    jw.endArray() catch return false;
    jw.endObject() catch return false;

    return sendResponse(alloc, sock, 200, "application/json", out.written(), req.origin, req.keep_alive);
}

fn handleOptions(alloc: Allocator, sock: SOCKET, req: HttpRequest) bool {
    if (req.origin) |orig| {
        if (!isAllowedOrigin(orig)) {
            return sendJsonError(alloc, sock, 403, "origin not allowed", "cors_error", null, null, req.keep_alive);
        }
    }
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(alloc);
    hdr.print(alloc, "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: {s}\r\n", .{
        if (req.keep_alive) "keep-alive" else "close",
    }) catch return false;
    appendCorsHeaders(alloc, &hdr, req.origin, true) catch return false;
    hdr.appendSlice(alloc, "\r\n") catch return false;
    return sendAll(sock, hdr.items);
}

// ============================================================================
// Per-connection worker thread
// ============================================================================

const ClientArgs = struct {
    state: *ServerState,
    sock: SOCKET,
};

fn handleClient(args: ClientArgs) void {
    const state = args.state;
    const sock = args.sock;
    defer {
        _ = shutdown(sock, SD_BOTH);
        _ = closesocket(sock);
    }

    const one: c_int = 1;
    _ = setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, @sizeOf(c_int));
    const rcv_timeout_ms: u32 = 30_000;
    _ = setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &rcv_timeout_ms, @sizeOf(u32));
    const snd_timeout_ms: u32 = 30_000;
    _ = setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &snd_timeout_ms, @sizeOf(u32));

    var conn: ConnBuf = .{
        .sock = sock,
        .alloc = state.alloc,
    };
    defer conn.deinit();

    while (true) {
        var req_arena = std.heap.ArenaAllocator.init(state.alloc);
        defer req_arena.deinit();
        const ra = req_arena.allocator();

        const req = readHttpRequest(&conn, ra) catch |err| switch (err) {
            error.Eof => return,
            error.PayloadTooLarge => {
                _ = sendJsonError(ra, sock, 413, "request body exceeds 32 MB limit", "invalid_request_error", "payload_too_large", null, false);
                return;
            },
            error.BadRequest => {
                _ = sendJsonError(ra, sock, 400, "malformed HTTP request", "invalid_request_error", null, null, false);
                return;
            },
            error.OutOfMemory => {
                _ = sendJsonError(ra, sock, 500, "out of memory", "server_error", null, null, false);
                return;
            },
        };

        var ok = true;
        if (std.mem.eql(u8, req.method, "OPTIONS")) {
            ok = handleOptions(ra, sock, req);
        } else if (std.mem.eql(u8, req.path, "/")) {
            if (std.mem.eql(u8, req.method, "GET")) {
                ok = sendResponse(ra, sock, 200, "text/html; charset=utf-8", UI_HTML, req.origin, req.keep_alive);
            } else {
                ok = sendJsonError(ra, sock, 405, "method not allowed", "invalid_request_error", null, req.origin, req.keep_alive);
            }
        } else if (std.mem.eql(u8, req.path, "/health")) {
            if (std.mem.eql(u8, req.method, "GET")) {
                ok = sendResponse(ra, sock, 200, "application/json", "{\"status\":\"ok\"}", req.origin, req.keep_alive);
            } else {
                ok = sendJsonError(ra, sock, 405, "method not allowed", "invalid_request_error", null, req.origin, req.keep_alive);
            }
        } else if (std.mem.eql(u8, req.path, "/v1/models")) {
            if (std.mem.eql(u8, req.method, "GET")) {
                ok = handleModels(state, ra, sock, req);
            } else {
                ok = sendJsonError(ra, sock, 405, "method not allowed", "invalid_request_error", null, req.origin, req.keep_alive);
            }
        } else if (std.mem.eql(u8, req.path, "/v1/chat/completions")) {
            if (std.mem.eql(u8, req.method, "POST")) {
                ok = handleChatCompletions(state, ra, sock, req);
            } else {
                ok = sendJsonError(ra, sock, 405, "method not allowed", "invalid_request_error", null, req.origin, req.keep_alive);
            }
        } else if (std.mem.eql(u8, req.path, "/v1/completions")) {
            if (std.mem.eql(u8, req.method, "POST")) {
                ok = handleCompletions(state, ra, sock, req);
            } else {
                ok = sendJsonError(ra, sock, 405, "method not allowed", "invalid_request_error", null, req.origin, req.keep_alive);
            }
        } else if (std.mem.eql(u8, req.path, "/tokenize")) {
            if (std.mem.eql(u8, req.method, "POST")) {
                ok = handleTokenize(state, ra, sock, req);
            } else {
                ok = sendJsonError(ra, sock, 405, "method not allowed", "invalid_request_error", null, req.origin, req.keep_alive);
            }
        } else {
            ok = sendJsonError(ra, sock, 404, "not found", "invalid_request_error", null, req.origin, req.keep_alive);
        }

        if (!ok or !req.keep_alive) return;
    }
}

// ============================================================================
// Server entry point
// ============================================================================

pub fn serve(alloc: std.mem.Allocator, engine: *Engine, host: []const u8, port: u16, model_name: []const u8) !void {
    var wsa_data: WsaDataBuf = undefined;
    if (WSAStartup(0x0202, &wsa_data) != 0) {
        return error.WsaStartupFailed;
    }
    defer _ = WSACleanup();

    const ip_addr = try parseIpv4Host(host);

    const listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_sock == INVALID_SOCKET) {
        return error.SocketCreateFailed;
    }
    defer _ = closesocket(listen_sock);

    const reuse: c_int = 1;
    _ = setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &reuse, @sizeOf(c_int));

    var addr: SockAddrIn = .{
        .sin_family = @intCast(AF_INET),
        .sin_port = htons(port),
        .sin_addr = ip_addr,
    };

    if (bind(listen_sock, &addr, @sizeOf(SockAddrIn)) == SOCKET_ERROR) {
        return error.BindFailed;
    }

    if (listen(listen_sock, 128) == SOCKET_ERROR) {
        return error.ListenFailed;
    }

    var freq: i64 = 0;
    _ = QueryPerformanceFrequency(&freq);

    var state = ServerState{
        .alloc = alloc,
        .engine = engine,
        .model_name = model_name,
        .created_ts = unixTimestampSeconds(),
        .qpc_freq = freq,
    };

    while (true) {
        const client_sock = accept(listen_sock, null, null);
        if (client_sock == INVALID_SOCKET) {
            continue;
        }
        const thread = std.Thread.spawn(.{ .stack_size = 2 * 1024 * 1024 }, handleClient, .{
            ClientArgs{ .state = &state, .sock = client_sock },
        }) catch {
            _ = closesocket(client_sock);
            continue;
        };
        thread.detach();
    }
}

// ============================================================================
// Unit tests
// ============================================================================

test "CORS origin matching" {
    try std.testing.expect(isAllowedOrigin("http://127.0.0.1:8080"));
    try std.testing.expect(isAllowedOrigin("http://127.0.0.1:18099"));
    try std.testing.expect(isAllowedOrigin("http://127.0.0.1"));
    try std.testing.expect(isAllowedOrigin("http://localhost:3000"));
    try std.testing.expect(isAllowedOrigin("http://localhost"));

    try std.testing.expect(!isAllowedOrigin("https://localhost:3000"));
    try std.testing.expect(!isAllowedOrigin("http://localhost.evil.com:3000"));
    try std.testing.expect(!isAllowedOrigin("http://127.0.0.1.evil.com"));
    try std.testing.expect(!isAllowedOrigin("http://127.0.0.2:8080"));
    try std.testing.expect(!isAllowedOrigin("http://evil.com"));
    try std.testing.expect(!isAllowedOrigin("http://localhost:99999"));
    try std.testing.expect(!isAllowedOrigin("http://localhost:8080/path"));
    try std.testing.expect(!isAllowedOrigin("null"));
}
