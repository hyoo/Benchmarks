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

# def gpu_start_2():
#     command_string = "nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -i 0 -l 1 > output_file.csv"
#     os.system(command_string)

import datetime

class GPUMonitorThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.proc_id = None

    def run(self):
        # gpu_start_2()
        self.proc_id = subprocess.Popen(
            ["nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -i 0 -l 1 > output_file.csv"],
            shell=True)


    def stop(self):
        # the problem with this right now is that the pid that is being terminated is pid -1 to the pid we need
        self.proc_id.terminate()

        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if 'nvidia-smi' in line:
                print(line)
                print('found nvidia-smi and am attempting to kill')
                pid = int(line.split(None, 1)[0])
                os.kill(pid, signal.SIGKILL)



class CPUPoll(threading.Thread):
    def __init__(self, prof_gpu):
        threading.Thread.__init__(self, name='CPUPoll')
        self.stop_event = threading.Event()
        self.prof_gpu = prof_gpu

    def run(self):
        print('Thread {thread} started'.format(thread=threading.current_thread()))

        # cpu_percents = []
        if self.prof_gpu:
            mt = GPUMonitorThread()
            mt.daemon = True
            mt.start()

        while not self.stop_event.is_set():
            cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()

            # this data is simply written to a log file
            aa = "".join([str(datetime.datetime.now()), '|', str(cpu_percent), '|', str(mem)])
            logger.info(aa)
            #
            # cpu_percents.append(aa)
            # print('thread is running')
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

        input_dict = args[0]
        prof_gpu = True
        if 'no_gpu_prof' in input_dict.keys():
            if input_dict['no_gpu_prof'] is True:
                prof_gpu = False

        # where I was initially entering tf's profileContext
        # with ProfileContext('tmp2/') as pctx:
        #     return f(*args, **kwds)

        with CPUPoll(prof_gpu) as cpu_poll:
            return f(*args, **kwds)
    return decorated
