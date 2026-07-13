from scapy.all import rdpcap
import math
import csv
import os


def extract_features(pcap_file, window_size):
    """
    Extract six discriminative features from a PCAP file per time window.
    Outputs raw features (no normalization) to CSV.

    Features:
        f1: total packet count
        f2: SYN ratio (SYN count / SYN-ACK count)
        f3: unique source IP count
        f4: max HTTP requests per source IP
        f5: Shannon entropy of source IP distribution
        f6: UDP packet count
    """
    # Validate input file
    if not os.path.isfile(pcap_file):
        print("[ERROR] File does not exist: " + pcap_file)
        return []

    print("[INFO] File exists: " + pcap_file)

    # Read PCAP
    packets = rdpcap(pcap_file)

    # Handle empty PCAP
    if len(packets) == 0:
        print("[WARN] Empty PCAP file: " + pcap_file)
        return []

    # Compute time windows
    first_packet_time = float(packets[0].time)
    last_packet_time = float(packets[-1].time)
    total_duration = last_packet_time - first_packet_time
    print("[INFO] Total Duration: " + str(total_duration) + "s")

    N_w = int(total_duration / window_size)
    print("[INFO] Number of windows: " + str(N_w))

    # Handle case where duration < window_size
    if N_w == 0:
        print("[WARN] Duration < window_size. No windows generated.")
        return []

    dataset = []

    for i in range(N_w):
        window_start = first_packet_time + i * window_size
        window_end = window_start + window_size
        window_packets = [p for p in packets if window_start <= float(p.time) < window_end]
        print("[INFO] Window " + str(i) + ": " + str(len(window_packets)) + " packets")

        # Initialize counters
        syn_count = 0
        synack_count = 0
        unique_IP = set()
        http_counts = {}
        udp_count = 0
        ip_packet_count = {}

        # Single pass through window packets
        for p in window_packets:
            # TCP flags
            if p.haslayer("TCP"):
                flags = p["TCP"].flags
                if flags == "S":
                    syn_count += 1
                elif flags == "SA":
                    synack_count += 1

            # IP features
            if p.haslayer("IP"):
                src = p["IP"].src
                unique_IP.add(src)
                ip_packet_count[src] = ip_packet_count.get(src, 0) + 1

            # HTTP requests
            if p.haslayer("TCP") and p.haslayer("IP"):
                payload = bytes(p["TCP"].payload)
                if b"GET" in payload or b"POST" in payload:
                    src = p["IP"].src
                    http_counts[src] = http_counts.get(src, 0) + 1

            # UDP count
            if p.haslayer("UDP"):
                udp_count += 1

        # Compute features
        f1 = len(window_packets)

        # Avoid division by zero for SYN ratio
        if synack_count == 0:
            synack_count = 1
        f2 = syn_count / synack_count

        f3 = len(unique_IP)

        f4 = max(http_counts.values()) if http_counts else 0

        # Shannon entropy
        f5 = 0.0
        if f1 > 0:
            for ip in unique_IP:
                count = ip_packet_count[ip]
                prob = count / f1
                if prob > 0:
                    f5 -= prob * math.log2(prob)

        f6 = udp_count

        # RAW FEATURES — no L2 normalization (normalization deferred to Phase 3 quantum pipeline)
        v_raw = [f1, f2, f3, f4, f5, f6]

        # Label: 0 = normal, 1 = attack
        label = 0 if "normal" in pcap_file.lower() else 1
        dataset.append(v_raw + [label])

    # Generate CSV path
    csv_filename = pcap_file.replace(".pcap", "_features.csv")
    csv_filename = csv_filename.replace("/pcap/", "/csv/")

    # Ensure output directory exists
    output_dir = os.path.dirname(csv_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print("[INFO] Created directory: " + output_dir)

    # Write CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in dataset:
            writer.writerow(row)

    print("[INFO] CSV saved: " + csv_filename)
    print("[INFO] Total rows: " + str(len(dataset)))
    return dataset


if __name__ == "__main__":
    extract_features("/root/ddos-framework/data/pcap/SYN_LOW_30s_20260622_115435.pcap", 5)