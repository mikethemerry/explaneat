import numpy as np
import time

import logging


def one_hot_encode(vals):
    width = max(vals)
    newVals = []
    for val in vals:
        blank = [0. for _ in range(width + 1)]
        blank[val] = 1.
        newVals.append(blank)
    return np.asarray(newVals)


class MethodTimer():
    def __init__(self, functionName=""):

        self.logger = logging.getLogger("experimenter.methodTimer")
        self.start = time.time()
        self.functionName = functionName
        msg = 'The function - {fname} - has just started at {time}'
        self.logger.info(msg.format(fname=self.functionName, time=self.start))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.start
        msg = 'The function - {fname} - took {time} seconds to complete'
        self.logger.info(msg.format(fname=self.functionName, time=runtime))
