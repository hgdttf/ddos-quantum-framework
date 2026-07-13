from scapy.all import sniff, IP, TCP, UDP
import psutil
import time
import math
import threading
import csv
import os
from datetime import datetime

# Detection thresholds calibrated for RAW feature magnitudes
# Normalization is deferred to Phase 3 quantum preprocessing pipeline
THRESHOLDS = [
    ("f1_total_packets", 200, ">"),
    ("f2_syn_ratio", 5.0, ">"),
    ("f3_unique_ips", 3, "<"),
    ("f4_max_http", 20, ">"),
    ("f5_entropy", 0.5, "<"),
    ("f6_udp_volume", 100, ">"),
]


def extract_window_features(window_packets):
    """
    Extract six discriminative features from a packet window.
    Returns raw features (no normalization).
    """
    f1 = len(window_packets)

    # SYN and SYN-ACK counting
    syn_count = 0
    synack_count = 0
    for p in window_packets:
        if p.haslayer("TCP"):
            flags = p["TCP"].flags
            if flags == "S":
                syn_count += 1
            elif flags == "SA":
                synack_count += 1

    # Avoid division by zero
    if synack_count == 0:
        synack_count = 1
    f2 = syn_count / synack_count

    # IP counting (single pass for f3 and f5)
    ip_counts = {}
    for p in window_packets:
        if p.haslayer("IP"):
            src = p["IP"].src
            ip_counts[src] = ip_counts.get(src, 0) + 1
    f3 = len(ip_counts)

    # HTTP request counting per source IP
    http_counts = {}
    for p in window_packets:
        if p.haslayer("TCP") and p.haslayer("IP"):
            payload = bytes(p["TCP"].payload)
            if b"GET" in payload or b"POST" in payload:
                src = p["IP"].src
                http_counts[src] = http_counts.get(src, 0) + 1
    f4 = max(http_counts.values()) if http_counts else 0

    # Shannon entropy of source IP distribution
    if f1 == 0:
        f5 = 0.0
    else:
        entropy = 0.0
        for count in ip_counts.values():
            prob = count / f1
            if prob > 0:
                entropy -= prob * math.log2(prob)
        f5 = entropy

    f6 = sum(1 for p in window_packets if p.haslayer("UDP"))

    return [f1, f2, f3, f4, f5, f6]


def detect(features, thresholds):
    """
    Compare raw features against calibrated thresholds.
    Returns: (is_attack, triggered_feature_names, confidence_percentage)
    """
    triggered = []
    for i, (name, threshold, direction) in enumerate(thresholds):
        value = features[i]
        if direction == ">" and value > threshold:
            triggered.append((name, value, threshold))
        elif direction == "<" and value < threshold:
            triggered.append((name, value, threshold))

    if not triggered:
        return False, [], 0.0

    # Confidence: average percentage deviation from thresholds
    confidence = 0.0
    for _, value, threshold in triggered:
        if threshold != 0:
            confidence += abs(value - threshold) / abs(threshold)
    confidence = (confidence / len(triggered)) * 100
    confidence = min(confidence, 100.0)

    return True, [t[0] for t in triggered], confidence


def sample_metrics(prev_net, attack_pid=None, attack_cpu_prev=None):
    """
    Sample current system metrics.
    Returns: (metrics_dict, updated_attack_cpu_prev)
    """
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    net_total = psutil.net_io_counters().bytes_recv + psutil.net_io_counters().bytes_sent
    net_rate = net_total - prev_net
    load_avg = os.getloadavg()[0]

    metrics = {
        "cpu": cpu,
        "mem": mem,
        "net": net_rate,
        "load": load_avg,
    }

    # Attack process CPU monitoring (optional)
    if attack_pid is not None and attack_cpu_prev is not None:
        try:
            attack_proc = psutil.Process(attack_pid)
            times_now = attack_proc.cpu_times()
            time_now = time.time()

            if attack_cpu_prev["time"] is not None and attack_cpu_prev["times"] is not None:
                dt = time_now - attack_cpu_prev["time"]
                d_cpu = (times_now.user + times_now.system) - (
                            attack_cpu_prev["times"].user + attack_cpu_prev["times"].system)
                if dt > 0:
                    metrics["attack_cpu"] = (d_cpu / dt) * 100.0
                else:
                    metrics["attack_cpu"] = 0.0
            else:
                metrics["attack_cpu"] = 0.0

            attack_cpu_prev = {"time": time_now, "times": times_now}

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            metrics["attack_cpu"] = 0.0
            attack_cpu_prev = {"time": None, "times": None}

        return metrics, attack_cpu_prev

    return metrics, attack_cpu_prev


def degradation(current, baseline):
    """
    Compute system degradation percentages.
    Returns: (rho_cpu, rho_mem, rho_net, rho_load, rho_composite)
    """

    def pct(c, b):
        if b == 0:
            return 0.0
        raw = ((c - b) / b) * 100
        return max(raw, -100.0)

    rho_cpu = pct(current["cpu"], baseline["cpu"])
    rho_mem = pct(current["mem"], baseline["mem"])
    rho_net = pct(current["net"], baseline["net"])
    rho_load = pct(current["load"], baseline["load"])

    # Composite: weighted average (net excluded due to near-zero baseline on loopback)
    rho = (rho_cpu * 0.3 + rho_mem * 0.3 + rho_load * 0.4)

    return rho_cpu, rho_mem, rho_net, rho_load, rho


def _sample(prev_net):
    """Helper: sample CPU, memory, and network rate."""
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    net_total = psutil.net_io_counters().bytes_recv + psutil.net_io_counters().bytes_sent
    net_rate = net_total - prev_net
    return {"cpu": cpu, "mem": mem, "net": net_rate}


def record_baseline(duration_seconds):
    """
    Record baseline system metrics over a duration.
    Returns: baseline_dict
    """
    print(f"[baseline] Recording for {duration_seconds}s...")
    psutil.cpu_percent(interval=None)  # Initialize CPU percent

    cpu_samples, mem_samples, net_samples = [], [], []
    prev_net = psutil.net_io_counters().bytes_recv + psutil.net_io_counters().bytes_sent
    start = time.time()

    while time.time() - start < duration_seconds:
        time.sleep(1)
        m = _sample(prev_net)
        cpu_samples.append(m["cpu"])
        mem_samples.append(m["mem"])
        net_samples.append(m["net"])
        prev_net = psutil.net_io_counters().bytes_recv + psutil.net_io_counters().bytes_sent

    load_avg = os.getloadavg()[0]

    return {
        "cpu": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
        "mem": sum(mem_samples) / len(mem_samples) if mem_samples else 0,
        "net": sum(net_samples) / len(net_samples) if net_samples else 0,
        "load": load_avg,
    }


def run_detection(duration, window_size, baseline_duration=10, interface="lo",
                  alert_log_path=None, correlation_path=None, attack_pid=None):
    """
    Entry point: record baseline then run detection.
    """
    attack_cpu_prev = {"time": None, "times": None}
    baseline = record_baseline(baseline_duration)
    return run_detection_with_baseline(duration, window_size, baseline, interface,
                                       alert_log_path, correlation_path, attack_pid, attack_cpu_prev)


def run_detection_with_baseline(duration, window_size, baseline, interface="lo",
                                alert_log_path=None, correlation_path=None,
                                attack_pid=None, attack_cpu_prev=None):
    """
    Run real-time detection with pre-recorded baseline.
    """
    if attack_cpu_prev is None:
        attack_cpu_prev = {"time": None, "times": None}

    # Default paths
    if alert_log_path is None:
        alert_log_path = os.path.join("/root", "ddos-framework", "results", "reports", "alert_log.csv")
    if correlation_path is None:
        correlation_path = os.path.join("/root", "ddos-framework", "results", "reports", "correlation_report.csv")

    print(
        f"[detect] Baseline: CPU={baseline['cpu']:.1f}% MEM={baseline['mem']:.1f}% NET={baseline['net']:.0f} B/s LOAD={baseline['load']:.2f}")

    # Thread-safe packet buffer
    packet_buffer = []
    buffer_lock = threading.Lock()
    alert_log = []
    correlation_report = []
    alert_count = 0

    def on_packet(pkt):
        """Callback: filter IP packets and append to buffer."""
        if pkt.haslayer("IP"):
            with buffer_lock:
                packet_buffer.append(pkt)

    # Start Scapy sniff in daemon thread
    sniff_thread = threading.Thread(
        target=lambda: sniff(iface=interface, prn=on_packet, store=False, timeout=duration),
        daemon=True,
    )
    sniff_thread.start()
    print(f"[detect] Capturing on '{interface}' for {duration}s. Window={window_size}s.")

    # Initialize sampling
    psutil.cpu_percent(interval=None)
    start = time.time()
    next_window = start + window_size
    next_sample = start + 1
    current = baseline.copy()
    prev_net = psutil.net_io_counters().bytes_recv + psutil.net_io_counters().bytes_sent

    # Main detection loop
    while time.time() - start <= duration:
        now = time.time()

        # Sample system metrics every second
        if now >= next_sample:
            current, attack_cpu_prev = sample_metrics(prev_net, attack_pid, attack_cpu_prev)
            prev_net = psutil.net_io_counters().bytes_recv + psutil.net_io_counters().bytes_sent
            next_sample = now + 1

        # Process window when interval expires
        if now >= next_window:
            with buffer_lock:
                window_packets = list(packet_buffer)
                packet_buffer.clear()

            # Extract raw features (no normalization — deferred to Phase 3 quantum pipeline)
            features = extract_window_features(window_packets)

            # Threshold-based detection on raw features
            is_attack, triggered, confidence = detect(features, THRESHOLDS)

            elapsed = now - start
            if is_attack:
                alert_count += 1
                rho_cpu, rho_mem, rho_net, rho_load, rho = degradation(current, baseline)
                ts = datetime.now().isoformat()

                alert_log.append({
                    "alert_id": alert_count,
                    "timestamp": ts,
                    "elapsed_s": round(elapsed, 2),
                    "f1": features[0], "f2": features[1], "f3": features[2],
                    "f4": features[3], "f5": features[4], "f6": features[5],
                    "triggered": ",".join(triggered),
                    "confidence_pct": round(confidence, 2),
                    "rho_cpu": round(rho_cpu, 2),
                    "attack_cpu": round(current.get("attack_cpu", 0), 2),
                    "sys_cpu": round(current["cpu"], 2),
                    "rho_mem": round(rho_mem, 2),
                    "rho_net": round(rho_net, 2),
                    "rho_load": round(rho_load, 2),
                    "rho_avg": round(rho, 2),
                })

                correlation_report.append({
                    "alert_id": alert_count,
                    "timestamp": ts,
                    "cpu_pct": current["cpu"],
                    "mem_pct": current["mem"],
                    "net_bytes": current["net"],
                    "load_avg": current["load"],
                    "rho_avg": round(rho, 2),
                })

                print(
                    f"[ALERT #{alert_count}] t={elapsed:.1f}s triggered={triggered} conf={confidence:.1f}% attack_cpu={current.get('attack_cpu', 0):.1f}% rho={rho:.1f}%")
            else:
                print(
                    f"[ok] t={elapsed:.1f}s f1={features[0]} f2={features[1]:.2f} f3={features[2]} f5={features[4]:.2f}")

            next_window = now + window_size

        # Sleep to prevent busy-waiting
        sleep_time = min(0.2, next_window - time.time(), next_sample - time.time())
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Wait for sniff thread to finish
    sniff_thread.join(timeout=5)

    # Export alert log
    if alert_log:
        os.makedirs(os.path.dirname(alert_log_path), exist_ok=True)
        with open(alert_log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=alert_log[0].keys())
            w.writeheader()
            w.writerows(alert_log)
        print(f"[done] Alerts: {len(alert_log)} -> {alert_log_path}")
    else:
        print("[done] No alerts triggered.")

    # Export correlation report
    if correlation_report:
        os.makedirs(os.path.dirname(correlation_path), exist_ok=True)
        with open(correlation_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=correlation_report[0].keys())
            w.writeheader()
            w.writerows(correlation_report)
        print(f"[done] Correlation: {len(correlation_report)} entries -> {correlation_path}")

    return alert_log, correlation_report


if __name__ == "__main__":
    # Test with synthetic baseline
    baseline = {
        "cpu": 5.0,
        "mem": 50.0,
        "net": 1000,
        "load": 0.5,
    }
    run_detection_with_baseline(30, 5, baseline, "lo")