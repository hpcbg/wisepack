#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# scripts/measure_dds_fiware_latency.py
#
# ADAPTED FROM HARMONY — scripts/measure_dds_fiware_latency.py (MIT, Copyright
# (c) 2026 Kaloyan Yovchev). The measurement method, the t0..t3 timestamp model,
# the "reference FIWARE->ROS from PATCH initiation" correction and the
# percentile/report shape are HARMONY's and are reused unchanged in substance.
#
# What differs: the topics and entities are WISEPACK's, and the PRIMARY metric is
# the ONE-WAY ROS -> FIWARE hop rather than the closed loop, because that is the
# hop the WISEPACK audit trail actually depends on. The closed loop is still
# measured and reported, since the operator command path needs FIWARE -> ROS.
#
#   ROS 2 publisher  (/wisepack/action/event, std_msgs/String)
#     -> Orion-LD DDS enabler maps it to
#        urn:ngsi-ld:WISEPACKActionStream:main  actionJson.value.data
#     -> this script PATCHes the operator command attribute (the minimal TEST
#        RELAY — measurement overhead, NOT workflow logic)
#        urn:ngsi-ld:WISEPACKSystem:main  command.value.data
#     -> Orion-LD writes it back to ROS 2 on /wisepack/operator/command
#     -> ROS 2 subscriber
#
# Per-sample timestamps (time.monotonic_ns):
#   t0   publish to the audit topic
#   t1   NGSI-LD attribute observed == payload   (ROS -> FIWARE  = t1 - t0)  <-- PRIMARY
#   t2   relay PATCH initiated
#   t2b  relay PATCH HTTP response received      (relay overhead = t2b - t2)
#   t3   payload received on the command topic   (FIWARE -> ROS  = t3 - t2)
#                                                 (loop          = t3 - t0)
#
# NOTE: the relay PATCH and the FIWARE -> ROS notification OVERLAP. Orion-LD
# writes to DDS while still processing the PATCH, so the subscriber can receive
# the message before the PATCH response returns (t3 < t2b). FIWARE -> ROS is
# therefore measured from PATCH *initiation* (t2), which never goes negative.
#
# stdlib + rclpy + std_msgs only.
# ---------------------------------------------------------------------------

import argparse
import csv
import json
import math
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception as exc:                                # pragma: no cover
    sys.stderr.write(
        "ERROR: could not import rclpy/std_msgs (%s).\n"
        "       Source a ROS 2 / Vulcanexus environment first, e.g.\n"
        "       source /opt/vulcanexus/jazzy/setup.bash\n" % exc)
    sys.exit(2)


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


ORION = os.environ.get("ORION", "http://localhost:1026").rstrip("/")
OUTDIR = os.environ.get("WISEPACK_RESULTS_DIR", "results")
WARMUP = env_int("DDS_LATENCY_WARMUP", 3)
SAMPLES = env_int("DDS_LATENCY_SAMPLES", 20)
TIMEOUT_SEC = env_int("DDS_LATENCY_TIMEOUT_SEC", 10)
POLL_INTERVAL_MS = env_int("DDS_LATENCY_POLL_INTERVAL_MS", 5)

INPUT_TOPIC = "/wisepack/action/event"
OUTPUT_TOPIC = "/wisepack/operator/command"
INPUT_ENTITY = "urn:ngsi-ld:WISEPACKActionStream:main"
INPUT_TYPE = "WISEPACKActionStream"
INPUT_ATTR = "actionJson"
OUTPUT_ENTITY = "urn:ngsi-ld:WISEPACKSystem:main"
OUTPUT_TYPE = "WISEPACKSystem"
OUTPUT_ATTR = "command"

_TTY = sys.stdout.isatty()
BOLD = "\033[1m" if _TTY else ""
CYN = "\033[36m" if _TTY else ""
GRN = "\033[32m" if _TTY else ""
YEL = "\033[33m" if _TTY else ""
RED = "\033[31m" if _TTY else ""
RST = "\033[0m" if _TTY else ""


def head(msg):
    print("\n%s%s%s" % (BOLD + CYN, msg, RST))
    print("-" * 60)


def _request(method, url, body=None, timeout=5):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception:
        return None, None


def broker_is_orionld():
    status, body = _request("GET", "%s/version" % ORION, timeout=3)
    if status is None or body is None:
        return None
    return b"orionld" in body.lower()


def read_input_attr():
    url = "%s/ngsi-ld/v1/entities/%s?local=true&attrs=%s" % (
        ORION, INPUT_ENTITY, INPUT_ATTR)
    status, body = _request("GET", url, timeout=3)
    if status != 200 or not body:
        return None
    try:
        d = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    a = d.get(INPUT_ATTR)
    if not isinstance(a, dict):
        return None
    v = a.get("value")
    return v.get("data") if isinstance(v, dict) else v


def ensure_entity(entity_id, entity_type, attr):
    url = "%s/ngsi-ld/v1/entities/%s?local=true" % (ORION, entity_id)
    status, _ = _request("GET", url, timeout=3)
    if status == 200:
        return True
    body = {"id": entity_id, "type": entity_type,
            attr: {"type": "Property", "value": {"data": "init"}}}
    status, _ = _request("POST", "%s/ngsi-ld/v1/entities" % ORION, body=body,
                         timeout=5)
    return status in (201, 204, 409)


def patch_output_attr(payload):
    """Relay onto the output attribute in the DDS nested-String shape.

    A plain `"value": "..."` does NOT propagate to DDS — the bridge requires
    `value.data`. HARMONY documents this; getting it wrong produces 204s and no
    ROS message, which looks like a latency failure and is not.
    """
    url = "%s/ngsi-ld/v1/entities/%s/attrs/%s" % (ORION, OUTPUT_ENTITY, OUTPUT_ATTR)
    body = {"type": "Property", "value": {"data": payload}}
    status, _ = _request("PATCH", url, body=body, timeout=TIMEOUT_SEC)
    return status in (204, 207)


class LoopProbe(Node):
    def __init__(self):
        super().__init__("wisepack_dds_fiware_latency_probe")
        self._pub = self.create_publisher(String, INPUT_TOPIC, 10)
        self._sub = self.create_subscription(String, OUTPUT_TOPIC,
                                             self._on_output, 10)
        self._lock = threading.Lock()
        self._expected = None
        self._received_ns = None
        self._event = threading.Event()

    def _on_output(self, msg):
        now = time.monotonic_ns()
        with self._lock:
            if self._expected is not None and self._expected in msg.data:
                self._received_ns = now
                self._event.set()

    def arm(self, payload):
        with self._lock:
            self._expected = payload
            self._received_ns = None
            self._event.clear()

    def publish_input(self, payload):
        self._pub.publish(String(data=payload))

    def wait_output(self, timeout):
        got = self._event.wait(timeout)
        with self._lock:
            return self._received_ns if got else None

    def input_subscription_count(self):
        return self._pub.get_subscription_count()

    def output_publisher_count(self):
        return self._sub.get_publisher_count()


def wait_for_discovery(node, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if node.input_subscription_count() >= 1 and node.output_publisher_count() >= 1:
            return True
        time.sleep(0.1)
    return False


def run_sample(node, index):
    # The probe payload is a well-formed ActionEvent so it exercises exactly the
    # real path, including the JSON size the real trail carries.
    marker = "PERF_%d_%d" % (time.monotonic_ns(), index)
    payload = json.dumps({
        "schema": "wisepack/1", "event_id": marker, "sequence": index,
        "stage": "IDLE", "action": "latency_probe", "actor": "system",
        "result": "ok", "source": "measured", "message": marker,
        "details": {"probe": True},
    }, separators=(",", ":"))
    poll_s = POLL_INTERVAL_MS / 1000.0

    node.arm(marker)
    t0 = time.monotonic_ns()
    node.publish_input(payload)

    t1 = None
    deadline = time.monotonic() + TIMEOUT_SEC
    while time.monotonic() < deadline:
        current = read_input_attr()
        if current and marker in str(current):
            t1 = time.monotonic_ns()
            break
        time.sleep(poll_s)
    if t1 is None:
        return {"index": index, "ok": False,
                "error": "audit attribute not observed (ROS->FIWARE timeout)"}

    t2 = time.monotonic_ns()
    if not patch_output_attr(marker):
        return {"index": index, "ok": False, "error": "relay PATCH rejected"}
    t2b = time.monotonic_ns()

    t3 = node.wait_output(TIMEOUT_SEC)
    if t3 is None:
        return {"index": index, "ok": False,
                "error": "command not received (FIWARE->ROS timeout)"}

    return {"index": index, "ok": True, "error": None,
            "loop_ns": t3 - t0, "ros_to_fiware_ns": t1 - t0,
            "patch_overhead_ns": t2b - t2, "fiware_to_ros_ns": t3 - t2}


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarise_ms(values_ns):
    vals = sorted(v / 1e6 for v in values_ns)
    return {"samples": len(vals), "min": vals[0],
            "median": statistics.median(vals), "mean": statistics.fmean(vals),
            "p95": percentile(vals, 95), "max": vals[-1]}


def fmt(x):
    return "%.3f ms" % x


def print_block(title, s):
    print("  %s:" % title)
    for key in ("samples", "min", "median", "mean", "p95", "max"):
        value = s[key]
        print("    %-8s %s" % (key + ":",
                               value if key == "samples" else fmt(value)))


def write_outputs(stamp, config, samples, summary, environment):
    os.makedirs(OUTDIR, exist_ok=True)
    json_path = os.path.join(OUTDIR, "wisepack-dds-fiware-latency-%s.json" % stamp)
    csv_path = os.path.join(OUTDIR, "wisepack-dds-fiware-latency-%s.csv" % stamp)
    timeouts = [s for s in samples if not s["ok"]]
    with open(json_path, "w") as f:
        json.dump({"timestamp": stamp, "config": config,
                   "environment": environment, "summary": summary,
                   "timeouts": len(timeouts), "samples": samples}, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "ok", "loop_ms", "ros_to_fiware_ms",
                    "patch_overhead_ms", "fiware_to_ros_ms", "error"])
        for s in samples:
            if s["ok"]:
                w.writerow([s["index"], 1,
                            "%.3f" % (s["loop_ns"] / 1e6),
                            "%.3f" % (s["ros_to_fiware_ns"] / 1e6),
                            "%.3f" % (s["patch_overhead_ns"] / 1e6),
                            "%.3f" % (s["fiware_to_ros_ns"] / 1e6), ""])
            else:
                w.writerow([s["index"], 0, "", "", "", "", s["error"]])
    return json_path, csv_path


def environment_facts():
    import platform
    return {"python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
            "rmw": os.environ.get("RMW_IMPLEMENTATION", "default (Fast DDS)"),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
            "vulcanexus": os.path.exists("/opt/vulcanexus")}


def main():
    argparse.ArgumentParser(
        description="Measure WISEPACK DDS->FIWARE audit-trail latency."
    ).parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config = {"orion": ORION, "warmup": WARMUP, "samples": SAMPLES,
              "timeout_sec": TIMEOUT_SEC, "poll_interval_ms": POLL_INTERVAL_MS,
              "input_topic": INPUT_TOPIC, "output_topic": OUTPUT_TOPIC,
              "input_entity": "%s.%s.value.data" % (INPUT_ENTITY, INPUT_ATTR),
              "output_entity": "%s.%s.value.data" % (OUTPUT_ENTITY, OUTPUT_ATTR),
              "primary_metric": "ros_to_fiware (the audit-trail hop)"}
    env = environment_facts()

    head("WISEPACK DDS -> FIWARE latency")
    print("    Orion-LD  : %s" % ORION)
    print("    audit hop : %s -> %s.%s" % (INPUT_TOPIC, INPUT_ENTITY, INPUT_ATTR))
    print("    command   : %s.%s -> %s" % (OUTPUT_ENTITY, OUTPUT_ATTR, OUTPUT_TOPIC))
    print("    warmup=%d samples=%d timeout=%ds poll=%dms"
          % (WARMUP, SAMPLES, TIMEOUT_SEC, POLL_INTERVAL_MS))
    print("    NOTE: the ROS->FIWARE figure INCLUDES this script's HTTP polling")
    print("          interval (%d ms), so it is an upper bound on the true"
          % POLL_INTERVAL_MS)
    print("          propagation delay, not a lower one.")

    is_ld = broker_is_orionld()
    if is_ld is None:
        print("  %sx%s Orion-LD not reachable at %s" % (RED, RST, ORION))
        return 2
    if not is_ld:
        print("  %sx%s broker at %s is not Orion-LD (NGSI-v2 stack?)" % (RED, RST, ORION))
        return 2
    print("  %s+%s Orion-LD (DDS/NGSI-LD) broker confirmed" % (GRN, RST))

    ensure_entity(INPUT_ENTITY, INPUT_TYPE, INPUT_ATTR)
    ensure_entity(OUTPUT_ENTITY, OUTPUT_TYPE, OUTPUT_ATTR)

    rclpy.init()
    node = LoopProbe()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=lambda: executor.spin(), daemon=True).start()

    try:
        if wait_for_discovery(node, TIMEOUT_SEC):
            print("  %s+%s DDS enabler matched on both topics" % (GRN, RST))
        else:
            print("  %s!%s DDS enabler not fully matched (in=%d out=%d); warmup will prime it"
                  % (YEL, RST, node.input_subscription_count(),
                     node.output_publisher_count()))

        head("Warmup (%d, not recorded)" % WARMUP)
        for i in range(WARMUP):
            r = run_sample(node, -(i + 1))
            print("    warmup %d/%d: %s"
                  % (i + 1, WARMUP, "ok" if r["ok"] else "FAILED: " + r["error"]))

        head("Sampling (%d)" % SAMPLES)
        samples = []
        for i in range(SAMPLES):
            r = run_sample(node, i)
            samples.append(r)
            if r["ok"]:
                print("    %2d/%d: R->F %s | relay %s | F->R %s | loop %s"
                      % (i + 1, SAMPLES,
                         fmt(r["ros_to_fiware_ns"] / 1e6),
                         fmt(r["patch_overhead_ns"] / 1e6),
                         fmt(r["fiware_to_ros_ns"] / 1e6),
                         fmt(r["loop_ns"] / 1e6)))
            else:
                print("    %2d/%d: %sFAILED%s — %s"
                      % (i + 1, SAMPLES, RED, RST, r["error"]))

        ok_samples = [s for s in samples if s["ok"]]
        if not ok_samples:
            print("\n  %sx%s No successful samples — broken DDS mapping or the "
                  "broker is not in DDS mode." % (RED, RST))
            json_path, csv_path = write_outputs(stamp, config, samples, {}, env)
            print("    wrote %s" % json_path)
            return 1

        summary = {
            "ros_to_fiware": summarise_ms([s["ros_to_fiware_ns"] for s in ok_samples]),
            "fiware_to_ros": summarise_ms([s["fiware_to_ros_ns"] for s in ok_samples]),
            "patch_overhead": summarise_ms([s["patch_overhead_ns"] for s in ok_samples]),
            "loop": summarise_ms([s["loop_ns"] for s in ok_samples]),
        }

        head("Results")
        print("\nPRIMARY — ROS 2 -> FIWARE (the audit-trail hop):")
        print_block("ros_to_fiware", summary["ros_to_fiware"])
        print("\nSecondary:")
        print_block("FIWARE -> ROS 2 (operator command path)", summary["fiware_to_ros"])
        print_block("closed loop (ROS -> FIWARE -> ROS)", summary["loop"])
        print_block("test-relay PATCH overhead (measurement cost, not workflow)",
                    summary["patch_overhead"])

        failed = len(samples) - len(ok_samples)
        if failed:
            print("\n  %s!%s %d/%d samples timed out — recorded in the artefact."
                  % (YEL, RST, failed, len(samples)))

        print("\nEnvironment: %s, ROS_DISTRO=%s, Vulcanexus=%s, domain=%s"
              % (env["platform"], env["ros_distro"], env["vulcanexus"],
                 env["ros_domain_id"]))

        json_path, csv_path = write_outputs(stamp, config, samples, summary, env)
        print("\nWrote:\n  %s\n  %s" % (json_path, csv_path))
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
