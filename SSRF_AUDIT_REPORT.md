# SSRF Security Audit Report

**Target:** EveryRow MCP Server (`everyrow-mcp/src/everyrow_mcp/`)
**Audit Date:** 2026-02-25
**Scope:** Full SSRF attack surface analysis, bypass testing of existing protections, container/deployment review
**Baseline:** Commit `4000b88` ("Security hardening: SSRF, headers, Redis TLS, container lockdown")

---

## Executive Summary

The MCP server implements a **three-layer SSRF protection** around its sole user-controlled URL-fetching path (`fetch_csv_from_url`), introduced in commit `4000b88`. The protections are well-designed and cover the major attack vectors. No **Critical** vulnerabilities were found. The two highest-priority findings — DNS-rebinding TOCTOU (FINDING-01) and missing port restrictions (FINDING-02) — have been **fixed** in this PR.

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 0 | — |
| High | 1 | **Fixed** (FINDING-01) |
| Medium | 3 | 1 **fixed** (FINDING-02), 2 open |
| Low | 4 | Hardening opportunities |
| Info | 4 | Defence-in-depth notes |

---

## Scope Limitations

This audit is based on **static analysis** of the source code. The following were not tested:

- No runtime DNS rebinding was attempted against a live instance
- No live container escape testing
- No fuzzing of URL parser edge cases
- No review of the EveryRow SDK client code (only the MCP server)
- No penetration testing of the deployed Kubernetes infrastructure

---

## Architecture Overview

### SSRF Attack Surface

The server has **two transport modes** with different trust boundaries:

| Mode | User-controlled URLs? | File system access? | Auth required? |
|------|----------------------|---------------------|---------------|
| stdio | Yes (URL + local path) | Yes (local CSV read/write) | No (API key) |
| HTTP | Yes (URL only) | No (blocked by validator) | Yes (OAuth 2.1) |

**Single user-controlled URL entry point:**

```
everyrow_upload_data(source=<user_url>)
  → UploadDataInput.validate_source()      # scheme check
    → validate_url()                        # http/https only
  → fetch_csv_from_url(url)                 # 3-layer SSRF protection
```

**Outbound HTTP clients (non-user-controlled):**

| Client | File:Line | Target | User-controlled? |
|--------|-----------|--------|-----------------|
| `fetch_csv_from_url` | `utils.py:188` | User-provided URL | **Yes** — SSRF-protected |
| `EveryRowAuthProvider._http_client` | `auth.py:199` | `settings.supabase_url` | No — config-derived |
| `SupabaseTokenVerifier._jwks_client` | `auth.py:53` | `{supabase_url}/.well-known/jwks.json` | No — config-derived |
| `AuthenticatedClient` (SDK) | `tool_helpers.py:65` | `settings.everyrow_api_url` | No — config-derived |
| `AuthenticatedClient` (upload) | `uploads.py:312` | `settings.everyrow_api_url` | No — config-derived |
| `AuthenticatedClient` (routes) | `routes.py:99` | `settings.everyrow_api_url` | No — config-derived |

---

## Existing SSRF Protections (Commit 4000b88)

### Layer 1: Pre-flight DNS Validation

**File:** `everyrow-mcp/src/everyrow_mcp/utils.py:53-100`

Before any HTTP request, `_validate_url_target(url)` extracts the hostname via `urlparse` and calls `_validate_hostname()`, which:

1. Checks against `_BLOCKED_HOSTNAMES` (`metadata.google.internal` + FQDN variant)
2. For IP literals: validates directly against `_BLOCKED_NETWORKS` with IPv4-mapped IPv6 unwrapping
3. For DNS names: resolves via `socket.getaddrinfo(AF_UNSPEC)` and checks every resolved IP

**Blocked networks (utils.py:21-31):**

| Network | Purpose |
|---------|---------|
| `10.0.0.0/8` | RFC 1918 private |
| `172.16.0.0/12` | RFC 1918 private |
| `192.168.0.0/16` | RFC 1918 private |
| `127.0.0.0/8` | Loopback |
| `169.254.0.0/16` | Link-local / cloud metadata |
| `0.0.0.0/8` | This network |
| `::1/128` | IPv6 loopback |
| `fc00::/7` | IPv6 ULA |
| `fe80::/10` | IPv6 link-local |

### Layer 2: Transport-Level IP Pinning

**File:** `everyrow-mcp/src/everyrow_mcp/utils.py:168-185`

Custom `_SSRFSafeTransport(httpx.AsyncBaseTransport)` resolves DNS, validates the resolved IPs, and **pins the connection** to the validated IP. The original hostname is preserved in the `Host` header and TLS SNI extension.

### Layer 3: Redirect Chain Validation

**File:** `everyrow-mcp/src/everyrow_mcp/utils.py:152-165`

An httpx `event_hooks["response"]` hook validates every redirect `Location` header against `_validate_url_target()` before following. Redirects are capped at `max_redirects=5`.

### Additional Controls

| Control | File:Line | Details |
|---------|-----------|---------|
| Scheme restriction | `utils.py:117` | Only `http://` and `https://` |
| Port allowlist | `utils.py:40-55` | Only 80, 443, 8080, 8443 |
| Streaming size limit | `utils.py:219-224` | 50 MB default, aborts mid-stream |
| Content-Length pre-check | `utils.py:212` | Rejects before streaming if header present |
| IPv4-mapped IPv6 unwrap | `utils.py:48-49, 68-70` | `::ffff:127.0.0.1` → `127.0.0.1` |
| Fail-closed on unparseable IPs | `utils.py:46` | Returns `True` (blocked) |
| Fail-closed on DNS failure | `utils.py:82` | Raises `ValueError` |

---

## Findings

### FINDING-01: DNS Rebinding TOCTOU Window — FIXED

**Severity:** High (residual risk after mitigation)
**File:** `everyrow-mcp/src/everyrow_mcp/utils.py:179-182`
**Status:** **Fixed** — transport now pins resolved IP

**Description:**

The original `_SSRFSafeTransport` re-validated hostnames at request time, but its `_validate_hostname()` call performed its own `socket.getaddrinfo()` lookup, which was a **separate DNS resolution** from what httpx's inner `AsyncHTTPTransport` (via `httpcore`) performed when opening the TCP connection. If DNS rebinds between the transport-level check and the actual `connect()` inside `httpcore`, a fast-rebinding DNS server could succeed.

```
Original timeline (vulnerable):
  T0: _validate_hostname() → getaddrinfo() → returns 93.184.216.34 (public) ✓
  T1: DNS rebinds hostname → 169.254.169.254 (metadata)
  T2: AsyncHTTPTransport → httpcore.connect() → getaddrinfo() → 169.254.169.254
  T3: Connection established to cloud metadata service
```

The window between T0 and T2 was extremely narrow (microseconds within the same async coroutine). Successful exploitation would require an attacker-controlled DNS server with very fast rebinding and is probabilistic — the actual success rate depends heavily on OS resolver caching behaviour and configured TTLs.

**Fix Applied:**

The transport now uses `_resolve_and_validate()` to resolve DNS once, validate all IPs, then pins the connection to the validated IP by rewriting the request URL. The original hostname is preserved in the `Host` header and TLS SNI extension (addressing the TLS/SNI regression risk noted in code review):

```python
# Resolve DNS and validate — returns the first safe IP
resolved_ip = _resolve_and_validate(hostname)

# Pin the URL to the validated IP
pinned_url = request.url.copy_with(host=resolved_ip)

# Preserve original hostname in Host header
headers = [...("host", hostname)...]

# Preserve original hostname for TLS SNI
extensions["sni_hostname"] = hostname.encode("ascii")
```

This eliminates the TOCTOU entirely — `httpcore` connects directly to the validated IP without performing a second DNS lookup.

---

### FINDING-02: No Port Restriction on Fetched URLs — FIXED

**Severity:** Medium
**File:** `everyrow-mcp/src/everyrow_mcp/utils.py:53-100`
**Status:** **Fixed** — port allowlist added

**Description:**

The SSRF blocklist validated IP addresses but did not restrict ports. An attacker could probe internal services on non-standard ports even when the IP is public or allowed. This is particularly dangerous in containerized deployments where services bind to non-standard ports.

**Proof of Concept:**

```python
# Probe Redis on its default port (if exposed on a public IP or shared network)
source = "http://redis.internal:6379/"

# SMTP banner grabbing
source = "http://mail.company.com:25/"
```

**Fix Applied:**

Added `_ALLOWED_PORTS = {80, 443, 8080, 8443}` and `_validate_port()`, enforced both in the pre-flight `_validate_url_target()` and at transport time in `_SSRFSafeTransport`. Non-allowed ports raise `ValueError`.

---

### FINDING-03: Incomplete IP Blocklist — Missing CGNAT Range

**Severity:** Medium
**File:** `everyrow-mcp/src/everyrow_mcp/utils.py:21-31`

**Description:**

The blocklist covers the major RFC 1918 ranges and cloud metadata endpoints but is missing `100.64.0.0/10` (RFC 6598 CGNAT), which is used by AWS for VPC endpoints and by Tailscale for mesh networking. An attacker could reach internal VPC services via this range.

| Missing Network | RFC | Severity |
|----------------|-----|----------|
| `100.64.0.0/10` | RFC 6598 (CGNAT) | **Medium** — reachable in AWS VPCs, GKE, Tailscale |
| `198.18.0.0/15` | RFC 2544 | Low — benchmark testing, sometimes used internally |
| `192.0.0.0/24` | RFC 6890 | Info — IANA special-purpose |
| `192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24` | RFC 5737 | Info — TEST-NET (documentation only, non-routable) |

**Proof of Concept:**

```python
# AWS VPC endpoint (CGNAT range)
source = "http://100.64.0.1/latest/meta-data/"

# Tailscale node in mesh network
source = "http://100.100.100.100/api/v1/status"
```

**Recommended Fix:**

```python
_BLOCKED_NETWORKS = [
    # ... existing entries ...
    ipaddress.ip_network("100.64.0.0/10"),    # CGNAT (RFC 6598) — AWS VPC, Tailscale
    ipaddress.ip_network("198.18.0.0/15"),     # Benchmark testing (RFC 2544)
    ipaddress.ip_network("192.0.0.0/24"),      # IANA special-purpose (RFC 6890)
]
```

---

### FINDING-04: Rate Limit Bypass via IP Header Spoofing

**Severity:** Medium
**File:** `everyrow-mcp/src/everyrow_mcp/middleware.py:28-40`

**Description:**

When `trust_proxy_headers=True`, `get_client_ip()` reads the first IP from `X-Forwarded-For` (or the configured header). If the reverse proxy does not strip or overwrite incoming `X-Forwarded-For` headers, an attacker can bypass rate limiting by spoofing different client IPs on each request.

```python
# middleware.py:36-39
if settings.trust_proxy_headers:
    value = request.headers.get(settings.trusted_ip_header.lower())
    if value:
        return value.split(",")[0].strip()  # Trusts first value
```

**Current Mitigation:** `docker-compose.yaml:42` defaults `TRUST_PROXY_HEADERS=false`. But Kubernetes/GKE deployments typically set this to `true`.

**Recommended Fix:**

1. Document that the reverse proxy MUST overwrite (not append to) the trusted IP header
2. Add the proxy's own IP to a `TRUSTED_PROXIES` allowlist and only read the header when the direct connection comes from a trusted proxy

---

### FINDING-05: Relative Redirect Blocking (False Positive / Fail-Safe)

**Severity:** Low
**File:** `everyrow-mcp/src/everyrow_mcp/utils.py:152-165`

**Description:**

The `_check_redirect` event hook validates the raw `Location` header from redirect responses. For relative redirects (e.g., `Location: /path`), `urlparse` returns `hostname=None`, causing `_validate_url_target` to raise `"URL has no hostname"`. This blocks the redirect chain.

From a security perspective, this is **fail-safe** — relative redirects stay on the same (already validated) host. However, it may cause false positives if a legitimate public server returns a relative redirect.

**Recommended Fix:**

Resolve relative redirects against the request URL before validating using `urljoin()`.

---

### FINDING-06: URL Parser Discrepancy (urlparse vs httpx)

**Severity:** Info
**File:** `everyrow-mcp/src/everyrow_mcp/utils.py:90-100`

**Description:**

The pre-flight validation uses Python's `urlparse` to extract the hostname, while httpx uses its own URL parser (via `httpcore`). Parser discrepancies could theoretically allow an attacker to craft a URL where `urlparse` extracts a different hostname than httpx.

**Mitigating Factor:** The `_SSRFSafeTransport` (Layer 2) re-validates using `request.url.host`, which is httpx's own parsed view, and then pins the connection to the resolved IP. This means the defence-in-depth is already working as designed — even if the pre-flight check (Layer 1) is fooled by a parser discrepancy, the transport-level check (Layer 2) uses the same parser that determines where the connection actually goes. The pre-flight check is purely an early-rejection optimisation.

---

### FINDING-07: Auth httpx Client Without SSRF Transport

**Severity:** Low
**File:** `everyrow-mcp/src/everyrow_mcp/auth.py:199-202`

**Description:**

The `EveryRowAuthProvider.__init__` creates an `httpx.AsyncClient()` without `_SSRFSafeTransport`. The target URL is derived from `settings.supabase_url`, which is a config value validated at startup to require HTTPS for non-localhost (`config.py:134-147`).

**Not exploitable** unless the attacker controls the server's environment variables.

**Recommended Fix (defence-in-depth):** Add the SSRF transport as a precaution.

---

### FINDING-08: Missing IPv6 Addresses in Blocklist

**Severity:** Low
**File:** `everyrow-mcp/src/everyrow_mcp/utils.py:21-31`

**Description:**

The blocklist does not include `::` (IPv6 unspecified address) or `::ffff:0:0/96` (IPv4-mapped prefix). The practical risk is minimal since `::` does not route to any reachable host in most environments.

**Recommended Fix:** Add `::/128` and `::ffff:0:0/96` to the blocklist.

---

### FINDING-09: `_decode_trusted_server_jwt` Skips Signature Verification

**Severity:** Info (by design)
**File:** `everyrow-mcp/src/everyrow_mcp/auth.py:152-161`

**Description:**

This function decodes JWTs with `verify_signature=False`. It is only called from `_issue_token_response` (line 456) with tokens obtained directly from Supabase's token endpoint over HTTPS. The critical guarantee that makes this safe is the HTTPS enforcement on `supabase_url` via the `_validate_url` validator in `config.py:134-147` — the token is received over a TLS-authenticated channel, making it trustworthy without a separate signature check.

**Assessment:** Safe by current usage. The docstring warning ("NEVER use this for tokens received from end users") is appropriate.

---

### FINDING-10: Wildcard CORS on Widget Endpoints

**Severity:** Info (safe by design)
**File:** `everyrow-mcp/src/everyrow_mcp/routes.py:22-33`

**Description:**

Widget endpoints use `Access-Control-Allow-Origin: *`. Since auth is via Bearer tokens (not cookies), this is safe per the CORS specification — no ambient credentials are leaked.

---

### FINDING-11: Container Hardening Review

**Severity:** Info
**File:** `everyrow-mcp/deploy/docker-compose.yaml`

**Description:**

The container configuration follows security best practices (non-root user, `no-new-privileges`, `cap_drop: ALL`, read-only rootfs, memory/CPU limits, network isolation, Redis password required, Redis not exposed to host, MCP bound to localhost).

**Missing (minor):** No `pids_limit` (fork bomb DoS), no `tmpfs` size limit.

---

## Attack Surface Matrix

| Attack Vector | Entry Point | Protection | Bypass? | Remediation |
|---------------|-------------|------------|---------|-------------|
| Direct SSRF via URL | `everyrow_upload_data(source=url)` | 3-layer validation + IP pinning | No (FINDING-01 fixed) | **Done** |
| Internal port probing | URL with non-standard port | Port allowlist (80, 443, 8080, 8443) | No (FINDING-02 fixed) | **Done** |
| SSRF via redirect | HTTP 3xx from attacker server | Event hook + transport re-check | Relative redirects blocked (fail-safe) | Open (Low) |
| Cloud metadata (169.254.169.254) | URL or DNS rebind | IP blocklist + hostname block | No | N/A |
| GKE metadata hostname | `metadata.google.internal` | Hostname blocklist | No | N/A |
| IPv4-mapped IPv6 bypass | `::ffff:127.0.0.1` | Unwrap + re-check | No | N/A |
| CGNAT range (100.64.x.x) | URL or DNS | **Not in blocklist** (FINDING-03) | **Yes** | Open (Medium) |
| Rate limit bypass | Spoofed X-Forwarded-For | Proxy trust config | Conditional (FINDING-04) | Open (Medium) |
| OAuth callback redirect | `/auth/callback` | Whitelist against registered URIs | No | N/A |
| File read via path | Local CSV path (stdio only) | Path validation + symlink resolve | No (HTTP mode rejects) | N/A |
| Redis key injection | Task IDs, user IDs | `build_key()` sanitization | No | N/A |
| Upload URL forgery | HMAC-SHA256 signature | `verify_upload_signature()` + expiry | No | N/A |

---

## Recommendations Summary

### Done (This PR)

1. **~~Fix DNS rebinding TOCTOU~~** (FINDING-01): Transport now pins resolved IPs — TOCTOU eliminated.
2. **~~Add port restrictions~~** (FINDING-02): Port allowlist `{80, 443, 8080, 8443}` enforced pre-flight and at transport time.

### Priority 2 (Medium Impact — Open)

3. **Expand IP blocklist** (FINDING-03): Add `100.64.0.0/10` (CGNAT), `198.18.0.0/15`, and `192.0.0.0/24`.
4. **Harden proxy IP trust** (FINDING-04): Validate the direct connection IP against a trusted proxy allowlist before reading forwarded headers.

### Priority 3 (Low Impact / Defence-in-Depth — Open)

5. **Resolve relative redirects** (FINDING-05): Use `urljoin()` to resolve relative Location headers before validating.
6. **Add SSRF transport to auth client** (FINDING-07): Defence-in-depth for `EveryRowAuthProvider._http_client`.
7. **Add IPv6 unspecified address** (FINDING-08): Block `::` and `::ffff:0:0/96`.
8. **Container hardening** (FINDING-11): Add `pids_limit`, `tmpfs` size limit.

---

## Conclusion

The SSRF protections introduced in commit `4000b88` represent a **well-engineered, defense-in-depth approach** that covers the major attack vectors. The three-layer architecture (pre-flight, transport, redirect) provides meaningful redundancy. The two highest-priority findings have been fixed in this PR: the DNS-rebinding TOCTOU is eliminated via IP pinning, and non-standard ports are now blocked via an allowlist. The remaining open items (CGNAT blocklist, proxy IP trust) are medium-priority hardening improvements. No critical vulnerabilities that would allow reliable SSRF exploitation were found.
