import functools
import os
import inspect

import multiprocessing as mp
import psutil
import subprocess

import threading
import time
import signal

import datetime
import pandas


def p2f(x):
    return float(x.strip('%'))


class GPUMonitorThread(threading.Thread):

    def __init__(self, output_dir):
        threading.Thread.__init__(self)
        self.proc_id = None
        self.output_dir = output_dir
        self.file_path = os.path.join(self.output_dir, 'profile.gpu.csv')

    def run(self):
        # save_path = os.path.join(self.output_dir, 'profile.gpu.csv')
        self.proc_id = subprocess.Popen(
            ["nvidia-smi --query-gpu=index,timestamp,utilization.gpu,utilization.memory --format=csv -l 1 > {}".format(
                self.file_path)],
            shell=True)

    def stop(self):
        # terminate the spawned proc
        self.proc_id.terminate()

        # still have to go and kill the nvidia-smi command, bc it is still logging
        p = subprocess.Popen(['ps', '-aux'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if 'nvidia-smi --query-gpu=index,timestamp,utilization.gpu' in str(line):
                print(str(line))
                print('found nvidia-smi and am attempting to kill')
                pid = int(str(line).split()[1])
                os.kill(pid, signal.SIGKILL)

        df = pandas.read_csv(self.file_path, converters={' utilization.memory [%]': p2f, ' utilization.gpu [%]': p2f})
        m = df.groupby(['index'])[' utilization.gpu [%]', ' utilization.memory [%]'].mean()
        with open(self.file_path, 'a+') as file:
            file.write(str(m))


class CPUPoll(threading.Thread):

    def __init__(self, prof_gpu, output_dir):
        threading.Thread.__init__(self, name='CPUPoll')
        self.stop_event = threading.Event()
        print('the value of prof_gpu is {}'.format(prof_gpu))
        self.prof_gpu = prof_gpu
        self.output_dir = output_dir
        self.file_path = os.path.join(output_dir, 'profile.cpu.csv')

        import logging
        self.logger = logging.getLogger('cpuutil')
        file_handler = logging.FileHandler(self.file_path, mode='w')
        # file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

    def run(self):

        print('Thread {thread} started'.format(thread=threading.current_thread()))

        if self.prof_gpu:
            mt = GPUMonitorThread(self.output_dir)
            mt.daemon = True
            mt.start()

        self.logger.info("timestamp, utilization.cpu[%], utilization.memory[%], utilization.memory[GB]")

        gig_val = 1024.0 ** 3

        while not self.stop_event.is_set():
            cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()

            # this data, cpu_percent and virtual memory, is written to a log file every 5 seconds
            aa = "".join([str(datetime.datetime.now()), ',', str(cpu_percent), ',', str(mem.percent), ',',
                          str(mem.used / gig_val)])
            self.logger.info(aa)
            time.sleep(5)

        if self.prof_gpu:
            mt.stop()

        print('Thread {thread} ended'.format(thread=threading.current_thread()))

    def stop(self):
        self.stop_event.set()
        df = pandas.read_csv(self.file_path)
        m = df.mean()
        with open(self.file_path, 'a+') as file:
            file.write(str(m))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()
        print('Force set Thread CPUPoll stop_event')


def run_profile(f):
    @functools.wraps(f)
    def decorated(*args, **kwds):

        # print(inspect.getmodule(f).__file__)

        # args[0] contains the overall run parameters
        output_dir = ''
        input_dict = args[0]
        if 'output_dir' in input_dict.keys():
            output_dir = input_dict['output_dir']

        # also can call 'which nvidia-smi' to see if it available
        cmd = ["which", "nvidia-smi"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        res = p.stdout.readlines()
        print(res)
        prof_gpu = False
        if len(res) > 0:
            print("nvidia-smi is there")
            prof_gpu = True

        # was initially entering tf's profileContext
        # could probably put that around the CPUPoll statement if we wanted

        # with ProfileContext('tmp2/') as pctx:
        #     return f(*args, **kwds)

        with CPUPoll(prof_gpu, output_dir) as cpu_poll:
            return f(*args, **kwds)

    return decorated


if __name__ == '__main__':
    # @run_profile
    # def dummy(dict_arg):
    #     pass
    #
    # dummy({})

    # True for gpu profiling, and output dir second path

    t = CPUPoll(True, '')
    t.start()
    time.sleep(10)
    t.stop()
