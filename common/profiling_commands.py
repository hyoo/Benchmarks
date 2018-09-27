import subprocess
import psutil
import datetime
import os


def gpu_start():
    command_string = ["nvidia-smi","--query-gpu=timestamp,utilization.gpu,utilization.memory", "--format=csv","-i", "0","-l", "1", "-f", "gpu_util.hostname.csv"]
    a = subprocess.Popen(command_string)

def gpu_start_2():
    command_string = "nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -i 0 -l 1 > output_file.csv"
    os.system(command_string)

def cpu_start():
    cpu_percent = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    print(datetime.datetime.now(),'|', cpu_percent,'|', mem)

# outputs to file
gpu_start_2()

# cpu_start()