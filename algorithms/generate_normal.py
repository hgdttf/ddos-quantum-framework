import requests
import time
import subprocess
import random
from datetime import datetime


def generate_normal_traffic():
    """
    Generates normal traffic PCAP with unique timestamp-based filename.
    Includes randomness in timing and endpoint selection for realistic variation.
    """
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    PCAP_PATH = f"/root/ddos-framework/data/pcap/NORMAL_30s_{timestamp}.pcap"

    # Verify target server is alive
    try:
        resp = requests.get("http://127.0.0.1:8080/", timeout=5)
        if resp.status_code != 200:
            print(f"ERROR: Target server returned {resp.status_code}")
            return None
    except Exception as e:
        print(f"ERROR: Cannot reach target server: {e}")
        return None

    # Start tshark capture (30s traffic + 2s buffer)
    tshark = subprocess.Popen(
        ["tshark", "-i", "lo", "-w", PCAP_PATH, "-a", "duration:32"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for tshark to initialize
    time.sleep(2)

    # Generate realistic browsing traffic with RANDOMNESS
    endpoints = ["/", "/about", "/login", "/contact"]
    start = time.time()
    packet_count = 0

    while time.time() - start < 30:
        # Randomize endpoint order each cycle
        random.shuffle(endpoints)

        for ep in endpoints:
            try:
                requests.get(f"http://127.0.0.1:8080{ep}", timeout=2)
                packet_count += 1
            except requests.exceptions.RequestException:
                pass
            # Random sleep between 0.1 and 0.5 seconds
            time.sleep(random.uniform(0.1, 0.5))

    # Force tshark to stop cleanly
    tshark.terminate()
    try:
        tshark.wait(timeout=5)
    except subprocess.TimeoutExpired:
        tshark.kill()
        tshark.wait()

    print(f"Normal traffic PCAP saved: {PCAP_PATH} ({packet_count} requests)")
    return PCAP_PATH


if __name__ == "__main__":
    generate_normal_traffic()