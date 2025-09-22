#!/usr/bin/env python3
import os
import json
import random
import re
import time
import shutil
import subprocess
import psutil
from datetime import datetime

# ------------- CONFIGURATION -------------
BASE_CONFIG = "ue.yaml"
UE_EXECUTABLE = "../build/nr-ue"

REWARD_CSV = "reward_metrics.csv"
Q_TABLE_FILE = "q_table.json"

TOTAL_EPISODES = 50
UES_PER_EPISODE = 25
MSG_RANGE = (1, 255)
STRESS_DURATION = 45  # seconds

# Valid parameters (maintaining 3GPP integrity)
ESTABLISHMENT_CAUSES = [
    "emergency",
    "highPriorityAccess",
    "mtAccess",
    "moSignalling",
    "moData"
]
SPARE_BITS = ["0", "1"]
VALID_PARAMS = list(range(1, 256))  # 1-255 as per spec

# RL parameters
EPSILON = 0.4
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.95
EPSILON_DECAY = 0.85

# ---------------- REWARD CONSTANTS ----------------
MAX_REWARD = 1000            # If the gNB has crashed
CRASH_BONUS_LATER = 200      # Extra reward if we had high load *before* crash

# (We remove negative penalty; we will return 0 for too-little usage)
W_CPU = 6                    # Weighted CPU usage
W_THREADS = 3                # Weighted thread count
W_NET = 2                    # Weighted net traffic
W_TIME = 1                   # Weighted “response/latency”

BASELINE_CPU = 2.0           # ~ CPU% for normal traffic
HIGH_LOAD_FACTOR = 4.0       # “dramatic” => 4× baseline => ~8% CPU
MIN_ACCEPTABLE_FACTOR = 1.5  # at least 1.5× baseline => ~3% CPU

os.umask(0)

# ------------- HELPER FUNCTIONS -------------
def get_gnb_pid():
    """Return PID of the gNB, or None if not running."""
    try:
        output = subprocess.check_output(["pgrep", "-f", "nr-gnb"])
        return int(output.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None

def get_gnb_metrics():
    """
    Returns (cpu%, thread_count).
    1) If gNB not found => treat as crash => (MAX_REWARD, 0)
    2) Prime CPU measurement to avoid reading 0.0% initially
    3) Clamp CPU between 1.0 and 100.0
    """
    pid = get_gnb_pid()
    if not pid:
        return (MAX_REWARD, 0)  # treat as crash

    try:
        proc = psutil.Process(pid)
        # Prime the CPU measurement
        proc.cpu_percent(interval=None)
        time.sleep(0.2)
        cpu_val = proc.cpu_percent(interval=None)
        if cpu_val < 1.0:
            cpu_val = 1.0
        if cpu_val > 100.0:
            cpu_val = 100.0

        threads = proc.num_threads()
        return (round(cpu_val, 1), threads)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return (MAX_REWARD, 0)

def generate_valid_config(episode, ue_id, cause, spare):
    """
    Creates a “valid” config with random parameters and returns (config_file, param, msg_count).
    """
    config_file = f"ue_config_{episode}_{ue_id}.yaml"
    
    try:
        # Ensure base config exists
        if not os.path.exists(BASE_CONFIG):
            raise FileNotFoundError(f"Base config {BASE_CONFIG} not found!")
        
        shutil.copy(BASE_CONFIG, config_file)
        
        imsi = f"00101{random.randint(1000000000, 9999999999)}"
        plmn = random.randint(100, 999)
        param = random.choice(VALID_PARAMS)
        msg_count = random.randint(*MSG_RANGE)

        with open(config_file, "r+") as f:
            content = f.read()
            content = re.sub(r"imsi:.*", f"imsi: {imsi}", content)
            content = re.sub(r"plmn_id:.*", f"plmn_id: {plmn}", content)
            content = re.sub(r"establishmentCause:.*", f"establishmentCause: {cause}", content)
            content = re.sub(r"spare:.*", f"spare: {spare}", content)
            content += f"\nfuzzedParam: {param}\nfuzzedMsgCount: {msg_count}\n"
            f.seek(0)
            f.write(content)
            f.truncate()

        return config_file, param, msg_count
    except Exception as e:
        print(f"Error generating config {config_file}: {str(e)}")
        return None, None, None  # Indicate failure

# ---------------- REWARD FUNCTION ----------------
def calculate_reward(cpu, threads, net_sent, net_recv, elapsed_time, crashed, was_high_load):
    """
    Summarize multiple signals => single reward, never negative.
    """
    if crashed:
        base = MAX_REWARD
        if was_high_load:
            base += CRASH_BONUS_LATER
        return base

    if cpu < (MIN_ACCEPTABLE_FACTOR * BASELINE_CPU):
        return 0.0

    cpu_part = W_CPU * cpu
    thread_part = W_THREADS * threads
    net_part = W_NET * (net_sent + net_recv) / 1e6
    time_part = W_TIME * (elapsed_time / 10.0)

    bonus = 0
    if cpu >= (HIGH_LOAD_FACTOR * BASELINE_CPU):
        bonus += 80

    raw = cpu_part + thread_part + net_part + time_part + bonus
    return min(raw, 2 * MAX_REWARD)

# ------------- Q-LEARNING AGENT -------------
class QLearningAgent:
    def __init__(self):
        self.q_table = self.load_q_table()
        self.epsilon = EPSILON

    def get_state_key(self, cpu, threads):
        return f"{cpu}_{threads}"

    def choose_action(self, state):
        if random.random() < self.epsilon or state not in self.q_table:
            cause = random.choice(ESTABLISHMENT_CAUSES)
            spare = random.choice(SPARE_BITS)
            return (cause, spare)

        best_action_key = max(self.q_table[state], key=self.q_table[state].get)
        return best_action_key.split("|", 1)

    def update_q_table(self, state, action, reward, next_state):
        state_actions = self.q_table.setdefault(state, {})
        old_q = state_actions.get(action, 0.0)
        next_max = max(self.q_table.get(next_state, {}).values(), default=0.0)

        new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_q)
        state_actions[action] = new_q
        self.epsilon *= EPSILON_DECAY

    def save_q_table(self):
        with open(Q_TABLE_FILE, "w") as f:
            json.dump(self.q_table, f)

    def load_q_table(self):
        if os.path.exists(Q_TABLE_FILE):
            try:
                with open(Q_TABLE_FILE, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

# ------------- MAIN EXECUTION -------------
def main():
    agent = QLearningAgent()
    if not get_gnb_pid():
        print("Start gNB first!")
        return

    with open(REWARD_CSV, "w") as f:
        f.write("timestamp,episode,cpu,threads,net_sent,net_recv,elapsed_time,reward,cause,spare\n")

    for episode in range(1, TOTAL_EPISODES + 1):
        if not get_gnb_pid():
            print(f"gNB crashed before episode {episode}!")
            break

        cpu_before, threads_before = get_gnb_metrics()

        net_start = psutil.net_io_counters()
        sent_start = net_start.bytes_sent
        recv_start = net_start.bytes_recv

        state_key = agent.get_state_key(cpu_before, threads_before)
        cause, spare = agent.choose_action(state_key)
        action_key = f"{cause}|{spare}"

        processes = []
        configs = []
        fuzz_params = []  # Track (param, msg_count)

        for ue_id in range(UES_PER_EPISODE):
            cfg, param, msg_count = generate_valid_config(episode, ue_id, cause, spare)
            if not cfg:  # Skip if config creation failed
                continue
            
            try:
                p = subprocess.Popen(
                    ["sudo", UE_EXECUTABLE, "-c", cfg],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                processes.append(p)
                configs.append(cfg)
                fuzz_params.append((param, msg_count))
            except Exception as e:
                print(f"Failed to start UE {ue_id}: {str(e)}")

        start_time = time.time()
        crashed = False
        was_high_load = False

        while True:
            elapsed = time.time() - start_time
            if elapsed >= STRESS_DURATION:
                break
            if not get_gnb_pid():
                crashed = True
                break

            cur_cpu, _ = get_gnb_metrics()
            if cur_cpu >= (HIGH_LOAD_FACTOR * BASELINE_CPU):
                was_high_load = True

            time.sleep(3)

        end_time = time.time()
        elapsed_time = end_time - start_time

        cpu_after, threads_after = get_gnb_metrics()
        net_end = psutil.net_io_counters()
        net_sent = net_end.bytes_sent - sent_start
        net_recv = net_end.bytes_recv - recv_start

        reward = calculate_reward(
            cpu_after, threads_after,
            net_sent, net_recv,
            elapsed_time, crashed,
            was_high_load
        )

        next_state = agent.get_state_key(cpu_after, threads_after)
        agent.update_q_table(state_key, action_key, reward, next_state)

        # Log CSV
        timestamp = datetime.now().isoformat()
        with open(REWARD_CSV, "a") as f:
            f.write(
                f"{timestamp},{episode},"
                f"{cpu_after},{threads_after},"
                f"{net_sent},{net_recv},"
                f"{elapsed_time:.2f},{reward:.2f},"
                f"{cause},{spare}\n"
            )

        # Cleanup processes and configs (only if no crash)
        for p in processes:
            p.terminate()

        if not crashed:
            for cfg in configs:
                try:
                    os.remove(cfg)
                except FileNotFoundError:
                    print(f"Config {cfg} not found during cleanup. Skipping...")
        else:
            crash_log = (
                f"\n=== CRASH DETECTED ===\n"
                f"Episode: {episode}\n"
                f"Action: cause={cause}, spare={spare}\n"
                f"Fuzzed Parameters:\n"
            )
            for i, (param, msg_count) in enumerate(fuzz_params):
                crash_log += f"  UE_{i}: param={param}, msg_count={msg_count}\n"
            
            print(crash_log)
            with open("crash_log.txt", "a") as f:
                f.write(crash_log)
            
            print(f"Crash configs preserved: {configs}")
            break  # Exit loop to preserve state

        print(
            f"[Ep {episode}] {cause}/{spare} | CPU={cpu_after}% Threads={threads_after} "
            f"NetSent={net_sent} NetRecv={net_recv} Time={elapsed_time:.2f}s "
            f"Reward={reward:.2f}"
        )

    agent.save_q_table()
    print("Fuzzing completed!")

if __name__ == "__main__":
    main()
