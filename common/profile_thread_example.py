# import tensorflow as tf
from tensorflow.contrib.tfprof import ProfileContext

import functools
import os
import inspect

import multiprocessing as mp
import psutil

import threading
import time

import logging
logger = logging.getLogger('cpuutil')
file_handler = logging.FileHandler('cpuutil.log', mode='w')
# file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)



import datetime

class Sleeper(threading.Thread):
    def __init__(self, sleep=5.0):
        threading.Thread.__init__(self, name='Sleeper')
        self.stop_event = threading.Event()
        self.sleep = sleep


        # logging.basicConfig(filename="cpuutil.log", level=logging.INFO)


    def run(self):
        print('Thread {thread} started'.format(thread=threading.current_thread()))
        cpu_percents = []

        while not self.stop_event.is_set():
            cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            # print(datetime.datetime.now(), '|', cpu_percent, '|', mem)
            aa = "".join([str(datetime.datetime.now()), '|', str(cpu_percent), '|', str(mem)])
            logger.info(aa)
            #
            # cpu_percents.append(aa)
            print('thread is running')
            time.sleep(5)
        print('Thread {thread} ended'.format(thread=threading.current_thread()))

    def stop(self):
        self.stop_event.set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()
        print('Force set Thread Sleeper stop_event')

# def cpu_percents():
#     cpu_percents = []
#     while worker_process.is_alive():
#         cpu_percents.append(p.cpu_percent())
#         time.sleep(0.01)
#
# def monitor(target):
#     worker_process = mp.Process(target=target)
#     worker_process.start()


def test_profile(f):
    @functools.wraps(f)
    def decorated(*args, **kwds):

        print(inspect.getmodule(f).__file__)

        if not os.path.isdir('tmp2'):
            os.mkdir('tmp2')

        # with ProfileContext('tmp2/') as pctx:
        #     return f(*args, **kwds)

        with Sleeper(sleep=2.0) as sleeper:
            return f(*args, **kwds)
    return decorated
