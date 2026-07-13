import os
import sys
import threading
import multiprocessing
import time
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify

sys.path.insert(0, "/root/ddos-framework/algorithms")

from Algorithm_1 import run_attack
from Algorithm_2 import extract_features
from Algorithm_3 import run_detection_with_baseline

app = Flask(__name__)

BASE_DIR = "/root/ddos-framework"
DATA_DIR = os.path.join(BASE_DIR, "data")
PCAP_DIR = os.path.join(DATA_DIR, "pcap")
DATASET_DIR = os.path.join(DATA_DIR, "datasets")
BASELINE_FILE = os.path.join(DATA_DIR, "baseline.json")

status_lock = threading.Lock()
attack_status = {
    "running": False,
    "result": None,
    "error": None,
    "params": None,
    "features_done": False,
    "features_csv": None,
    "dataset": None
}

auto_detection_status = {
    "running": False,
    "alerts": [],
    "start_time": None,
    "result": None,
    "error": None,
    "detection_id": None
}

baseline_data = None
baseline_ready = threading.Event()


def get_datasets():
    if not os.path.exists(DATASET_DIR):
        return []
    return sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(".pcap")])


def record_baseline(duration):
    import psutil
    psutil.cpu_percent(interval=None)
    cpu_samples, mem_samples, net_samples = [], [], []
    prev_net = psutil.net_io_counters().bytes_recv + psutil.net_io_counters().bytes_sent
    start = time.time()
    while time.time() - start < duration:
        time.sleep(1)
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        net_total = psutil.net_io_counters().bytes_recv + psutil.net_io_counters().bytes_sent
        net_rate = net_total - prev_net
        prev_net = net_total
        cpu_samples.append(cpu)
        mem_samples.append(mem)
        net_samples.append(net_rate)
    # load average is already a 1-minute rolling metric
    # averaging it again over the recording window contaminates it with warm-up noise
    # use the final reading — it reflects the current idle state
    load_avg = os.getloadavg()[0]
    return {
        "cpu": sum(cpu_samples) / len(cpu_samples),
        "mem": sum(mem_samples) / len(mem_samples),
        "net": sum(net_samples) / len(net_samples),
        "load": load_avg,
    }


def load_or_record_initial_baseline():
    global baseline_data
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r") as f:
            baseline_data = json.load(f)
        print(f"[startup] Loaded baseline: CPU={baseline_data['cpu']:.1f}% MEM={baseline_data['mem']:.1f}% LOAD={baseline_data['load']:.2f}")
        baseline_ready.set()
        return

    # load average is a 1-minute rolling metric
    # measuring too early captures Flask startup noise, not idle baseline
    print("[startup] Warming up 15s before baseline...")
    time.sleep(15)

    print("[startup] Recording 10s baseline...")
    baseline_data = record_baseline(10)

    os.makedirs(os.path.dirname(BASELINE_FILE), exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(baseline_data, f)

    print(f"[startup] Baseline saved: CPU={baseline_data['cpu']:.1f}% MEM={baseline_data['mem']:.1f}% LOAD={baseline_data['load']:.2f}")
    baseline_ready.set()


def run_attack_process(attack_type, intensity, target, duration, port, pcap_file, result_queue):
    pid = os.getpid()
    print(f"[attack] Process started. PID={pid}")
    try:
        pcap_filename, log = run_attack(attack_type, intensity, target, duration, port, pcap_file=pcap_file)
        result_queue.put({"success": True, "pid": pid, "pcap_file": pcap_filename, "log": log})
    except Exception as e:
        result_queue.put({"success": False, "pid": pid, "error": str(e)})


def run_attack_background(attack_type, intensity, target, duration, port, pcap_file=None):
    with status_lock:
        attack_status["running"] = True
        attack_status["result"] = None
        attack_status["error"] = None
        attack_status["params"] = {
            "type": attack_type, "intensity": intensity,
            "target": target, "duration": duration, "port": port,
            "pcap_file": pcap_file
        }
        attack_status["features_done"] = False
        attack_status["features_csv"] = None
        attack_status["dataset"] = []

    result_queue = multiprocessing.Queue()
    attack_proc = multiprocessing.Process(
        target=run_attack_process,
        args=(attack_type, intensity, target, duration, port, pcap_file, result_queue)
    )
    attack_proc.start()
    attack_pid = attack_proc.pid

    attack_proc.join()

    try:
        result = result_queue.get(timeout=5)
    except:
        result = {"success": False, "error": "Attack process did not return result"}

    if result["success"]:
        pcap_file_result = result["pcap_file"]
        pcap_path = os.path.join(PCAP_DIR, pcap_file_result)
        csv_file = pcap_path.replace(".pcap", "_features.csv").replace("/pcap/", "/csv/")

        try:
            dataset = extract_features(pcap_path, 5)
            features_success = True
        except Exception as e2:
            features_success = False
            dataset = []
            print(f"Feature extraction failed: {e2}")

        with status_lock:
            attack_status["result"] = pcap_file_result
            attack_status["running"] = False
            attack_status["features_done"] = features_success
            attack_status["features_csv"] = csv_file if features_success else None
            attack_status["dataset"] = dataset if features_success else []
    else:
        with status_lock:
            attack_status["error"] = result.get("error", "Unknown error")
            attack_status["running"] = False

    return attack_pid


def run_detection_background(duration, window_size, detection_id, attack_pid_queue, baseline):
    global auto_detection_status
    auto_detection_status = {
        "running": True,
        "alerts": [],
        "start_time": datetime.now().isoformat(),
        "result": None,
        "error": None,
        "detection_id": detection_id
    }

    try:
        attack_pid = None
        try:
            attack_pid = attack_pid_queue.get(timeout=2)
            print(f"[detect] Received attack PID={attack_pid}")
        except:
            print("[detect] No attack PID received")

        alert_log, correlation_report = run_detection_with_baseline(
            duration=duration,
            window_size=window_size,
            baseline=baseline,
            interface="lo",
            attack_pid=attack_pid
        )
        auto_detection_status["alerts"] = alert_log
        auto_detection_status["result"] = {
            "alert_count": len(alert_log),
            "correlation_count": len(correlation_report)
        }
        auto_detection_status["running"] = False
    except Exception as e:
        auto_detection_status["error"] = str(e)
        auto_detection_status["running"] = False


@app.route("/")
def home():
    baseline_is_ready = baseline_ready.is_set()
    return render_template("index.html", baseline_ready=baseline_is_ready)


@app.route("/attack", methods=["GET", "POST"])
def attack():
    datasets = get_datasets()

    if request.method == "POST":
        attack_type = request.form["attack_type"]
        intensity = request.form["intensity"]
        user_duration = int(request.form["duration"])
        target = "127.0.0.1"
        port = 8080

        selected_dataset = request.form.get("dataset_file", "")
        pcap_file = None
        if selected_dataset and selected_dataset in datasets:
            pcap_file = os.path.join(DATASET_DIR, selected_dataset)

        detection_duration = user_duration + 5
        attack_duration = user_duration

        print(f"[app] Attack duration={attack_duration}s, Detection duration={detection_duration}s")

        detection_id = "attack_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        attack_pid_queue = multiprocessing.Queue()

        detection_thread = threading.Thread(
            target=run_detection_background,
            args=(detection_duration, 5, detection_id, attack_pid_queue, baseline_data)
        )
        detection_thread.start()

        def attack_wrapper():
            result_queue = multiprocessing.Queue()
            attack_proc = multiprocessing.Process(
                target=run_attack_process,
                args=(attack_type, intensity, target, attack_duration, port, pcap_file, result_queue)
            )
            attack_proc.start()
            attack_pid = attack_proc.pid
            attack_pid_queue.put(attack_pid)
            print(f"[app] Sent attack PID={attack_pid} to detection")

            attack_proc.join()

            try:
                result = result_queue.get(timeout=5)
            except:
                result = {"success": False, "error": "No result from attack process"}

            if result["success"]:
                pcap_file_result = result["pcap_file"]
                pcap_path = os.path.join(PCAP_DIR, pcap_file_result)
                csv_file = pcap_path.replace(".pcap", "_features.csv").replace("/pcap/", "/csv/")

                try:
                    dataset = extract_features(pcap_path, 5)
                    features_success = True
                except Exception as e2:
                    features_success = False
                    dataset = []
                    print(f"Feature extraction failed: {e2}")

                with status_lock:
                    attack_status["result"] = pcap_file_result
                    attack_status["running"] = False
                    attack_status["features_done"] = features_success
                    attack_status["features_csv"] = csv_file if features_success else None
                    attack_status["dataset"] = dataset if features_success else []
            else:
                with status_lock:
                    attack_status["error"] = result.get("error", "Unknown error")
                    attack_status["running"] = False

        attack_wrapper_thread = threading.Thread(target=attack_wrapper)
        attack_wrapper_thread.start()

        return redirect(url_for("attack_status_page"))

    return render_template("attack.html", datasets=datasets)


@app.route("/attack/status")
def attack_status_page():
    with status_lock:
        status = dict(attack_status)

    if not status["running"] and status["result"] and status["features_done"]:
        return redirect(url_for("features_result_auto",
                                pcap_file=status["result"],
                                csv_file=status["features_csv"]))

    return render_template("attack_status.html", status=status)


@app.route("/api/attack/status")
def api_attack_status():
    with status_lock:
        return jsonify(dict(attack_status))


@app.route("/features/result")
def features_result_auto():
    pcap_file = request.args.get("pcap_file", "")
    csv_file = request.args.get("csv_file", "")

    with status_lock:
        dataset = attack_status.get("dataset", [])

    return render_template("features_result.html",
                           dataset=dataset,
                           pcap_file=pcap_file,
                           csv_file=csv_file,
                           auto_detection=True)


@app.route("/detection/result")
def detection_result():
    return render_template("detection_result.html", status=auto_detection_status)


@app.route("/api/detection/status")
def api_detection_status():
    return jsonify(dict(auto_detection_status))


@app.route("/features", methods=["GET", "POST"])
def features():
    if request.method == "POST":
        pcap_file = request.form["pcap_file"]
        window_size = int(request.form["window_size"])
        try:
            dataset = extract_features(pcap_file, window_size)
            return render_template("features_result.html",
                                   dataset=dataset,
                                   pcap_file=pcap_file,
                                   csv_file=pcap_file.replace(".pcap", "_features.csv").replace("/pcap/", "/csv/"),
                                   auto_detection=False)
        except Exception as e:
            return render_template("features.html", error=str(e))
    return render_template("features.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        duration = int(request.form["duration"])
        window_size = int(request.form["window_size"])

        dummy_queue = multiprocessing.Queue()
        t = threading.Thread(
            target=run_detection_background,
            args=(duration, window_size, "manual_" + datetime.now().strftime("%Y%m%d_%H%M%S"), dummy_queue, baseline_data)
        )
        t.start()

        return redirect(url_for("detect_status_page"))

    return render_template("detect.html")


@app.route("/detect/status")
def detect_status_page():
    return render_template("detect_status.html", status=auto_detection_status)


@app.route("/results")
def results():
    pcap_files = []
    csv_files = []

    if os.path.exists(PCAP_DIR):
        pcap_files = [f for f in os.listdir(PCAP_DIR) if f.endswith(".pcap")]
    csv_dir = os.path.join(DATA_DIR, "csv")
    if os.path.exists(csv_dir):
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    return render_template("results.html", pcap_files=pcap_files, csv_files=csv_files)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    baseline_thread = threading.Thread(target=load_or_record_initial_baseline, daemon=True)
    baseline_thread.start()

    print("[startup] Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=False)