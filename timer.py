#timer
import time


class timer:
    def __init__(self):
        self._start_time = None
    
    def start(self):
        self._start_time = time.perf_counter()
        return

    def end(self):
        elap_time = (time.perf_counter() - self._start_time)
        mins = elap_time / 60
        seconds = (elap_time % int(elap_time)) * 60
        print("Total elapsed time: {} mins {:0.4f} seconds".format(int(mins), seconds))
        return

