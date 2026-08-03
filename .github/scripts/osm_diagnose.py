"""Probe Overpass connectivity and report via GitHub annotations.

Temporary diagnostic for the macos-arm Overpass failure. Delete once resolved.
"""
import errno
import json
import platform
import socket
import ssl
import sys
import time
import urllib.request

TIMEOUT = 20
HOSTS = [
    "overpass-api.de",
    "overpass.kumi.systems",
    "overpass.private.coffee",
    "overpass.osm.ch",
]
QUERY = '[out:json][timeout:25];node(-34.3900,148.4650,-34.3850,148.4700);out count;'

lines = []


def log(msg):
    print(msg, flush=True)
    lines.append(msg)


def egress_ip(url):
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            return r.read().decode().strip()
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def describe_exc(e):
    """Surface the errno that the macOS failure reports (Errno 61 = ECONNREFUSED)."""
    eno = getattr(e, "errno", None)
    inner = getattr(e, "reason", None)
    if eno is None and inner is not None:
        eno = getattr(inner, "errno", None)
    name = errno.errorcode.get(eno, "") if eno else ""
    return f"{type(e).__name__}(errno={eno} {name}): {e}"


def tcp_probe(host, family, addr):
    s = socket.socket(family, socket.SOCK_STREAM)
    s.settimeout(TIMEOUT)
    t0 = time.time()
    try:
        s.connect(addr)
        return f"OK in {time.time() - t0:.1f}s"
    except Exception as e:
        return f"FAIL after {time.time() - t0:.1f}s {describe_exc(e)}"
    finally:
        s.close()


def https_get(url, force_v4=False):
    """GET url, optionally pinning resolution to IPv4 to test an IPv6-only failure."""
    orig = socket.getaddrinfo
    if force_v4:
        socket.getaddrinfo = lambda h, p, f=0, *a, **k: orig(h, p, socket.AF_INET, *a, **k)
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "shelterbelts-ci-diagnostic"})
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ssl.create_default_context()) as r:
            body = r.read(200).decode(errors="replace").replace("\n", " ")[:120]
            return f"HTTP {r.status} in {time.time() - t0:.1f}s | {body}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} in {time.time() - t0:.1f}s"
    except Exception as e:
        return f"FAIL after {time.time() - t0:.1f}s {describe_exc(e)}"
    finally:
        socket.getaddrinfo = orig


def overpass_query(host):
    url = f"https://{host}/api/interpreter"
    t0 = time.time()
    try:
        req = urllib.request.Request(
            url, data=QUERY.encode(), headers={"User-Agent": "shelterbelts-ci-diagnostic"}
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            body = r.read(300).decode(errors="replace").replace("\n", " ")[:150]
            return f"HTTP {r.status} in {time.time() - t0:.1f}s | {body}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} in {time.time() - t0:.1f}s | {e.read(200).decode(errors='replace')[:150]}"
    except Exception as e:
        return f"FAIL after {time.time() - t0:.1f}s {describe_exc(e)}"


log(f"== platform: {platform.platform()} machine={platform.machine()} python={sys.version.split()[0]}")
log(f"== has_ipv6={socket.has_ipv6}")
log(f"== egress v4: {egress_ip('https://api.ipify.org')}")
log(f"== egress v6: {egress_ip('https://api6.ipify.org')}")

for host in HOSTS:
    log(f"-- {host}")
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except Exception as e:
        log(f"   DNS FAIL {describe_exc(e)}")
        continue
    seen = []
    for family, _, _, _, sockaddr in infos:
        key = (family, sockaddr[0])
        if key in seen:
            continue
        seen.append(key)
        fam = "v6" if family == socket.AF_INET6 else "v4"
        log(f"   DNS {fam} {sockaddr[0]} -> tcp443 {tcp_probe(host, family, sockaddr)}")
    log(f"   GET /api/status default : {https_get(f'https://{host}/api/status')}")
    log(f"   GET /api/status ipv4only: {https_get(f'https://{host}/api/status', force_v4=True)}")
    log(f"   POST /api/interpreter   : {overpass_query(host)}")

report = "\n".join(lines)
# Annotations are the only channel readable without auth, and workflow commands
# need newlines percent-encoded to survive as a single annotation.
encoded = report.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
print(f"::error title=osm-diagnostic-{platform.machine()}::{encoded}", flush=True)

with open("osm_diagnostic.json", "w") as f:
    json.dump({"machine": platform.machine(), "report": report}, f)
