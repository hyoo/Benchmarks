import functools
import os
import inspect

import multiprocessing as mp
import psutil
import subprocess

import threading
import time
import signal


import logging
logger = logging.getLogger('cpuutil')
file_handler = logging.FileHandler('cpuutil.log', mode='w')
# file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


import datetime

class GPUMonitorThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.proc_id = None

    def run(self):

        self.proc_id = subprocess.Popen(
            ["nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -i 0 -l 1 > output_file.csv"],
            shell=True)


    def stop(self):
        # terminate the spawned proc
        self.proc_id.terminate()

        # still have to go and kill the nvidia-smi command, bc it is still logging
        p = subprocess.Popen(['ps', '-aux'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if 'nvidia-smi' in str(line):
                print(str(line))
                print('found nvidia-smi and am attempting to kill')
                pid = int(str(line).split()[1])
                os.kill(pid, signal.SIGKILL)



class CPUPoll(threading.Thread):
    def __init__(self, prof_gpu):
        threading.Thread.__init__(self, name='CPUPoll')
        self.stop_event = threading.Event()
        print('the value of prof_gpu is {}'.format(prof_gpu))
        self.prof_gpu = prof_gpu

    def run(self):
        print('Thread {thread} started'.format(thread=threading.current_thread()))


        if self.prof_gpu:
            mt = GPUMonitorThread()
            mt.daemon = True
            mt.start()

        while not self.stop_event.is_set():
            cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()

            # this data, cpu_percent and virtual memory, is written to a log file every 5 seconds
            aa = "".join([str(datetime.datetime.now()), '|', str(cpu_percent), '|', str(mem)])
            logger.info(aa)
            time.sleep(5)

        if self.prof_gpu:
            mt.stop()

        print('Thread {thread} ended'.format(thread=threading.current_thread()))

    def stop(self):
        self.stop_event.set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()
        print('Force set Thread CPUPoll stop_event')


def test_profile(f):
    @functools.wraps(f)
    def decorated(*args, **kwds):

        print(inspect.getmodule(f).__file__)

        if not os.path.isdir('tmp2'):
            os.mkdir('tmp2')

        # could also get the param from the command line
        input_dict = args[0]
        prof_gpu = True
        if 'no_gpu_prof' in input_dict.keys():
            if input_dict['no_gpu_prof'] is True:
                prof_gpu = False

        # also can call 'which nvidia-smi' to see if it available
        cmd = ["which", "nvidia-smi"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        res = p.stdout.readlines()
        print(res)
        prof_gpu = False
        if len(res) > 0:
            print("nvidia-smi is there")
            prof_gpu = True


        # where I was initially entering tf's profileContext
        # with ProfileContext('tmp2/') as pctx:
        #     return f(*args, **kwds)

        with CPUPoll(prof_gpu) as cpu_poll:
            return f(*args, **kwds)
    return decorated

if __name__ == '__main__':
    t = CPUPoll(True)
    t.start()
    time.sleep(15)
    t.stop()