import pandas as pd
import os
import uuid
import json
import copy
import datetime

"value",
"measurementType",
"experimentName",
"dataset",
"experimentSha",
"iteration"


class Result():
    def __init__(self,
                 value,
                 measurementType,
                 experimentName,
                 dataset,
                 experimentSha,
                 iteration,
                 params=None,
                 my_time=None,
                 my_id=None):

        self.value = value
        self.measurementType = measurementType
        self.experimentName = experimentName
        self.dataset = dataset
        self.experimentSha = experimentSha
        self.iteration = iteration
        self.params = params  # don't mind null params
        self.datetime = my_time if my_time is not None else datetime.datetime.now()
        self.my_id = my_id if my_id is not None else uuid.uuid4().hex

    def to_pd_record(self):
        record = pd.DataFrame([{
            "_id": self.my_id,
            "value": self.value,
            "measurementType": self.measurementType,
            "experimentName": self.experimentName,
            "dataset": self.dataset,
            "experimentSha": self.experimentSha,
            "iteration": self.iteration,
            "params": self.params,
            "datetime": self.datetime
        }])

        return record


class ResultsDatabase():
    INDEX = "_id"
    COLUMNS = [
        "_id",
        "value",
        "measurementType",
        "experimentName",
        "dataset",
        "experimentSha",
        "iteration",
        "params",
        "datetime"

    ]

    def __init__(self,
                 filePath,
                 saveOnDestroy=True):
        self.init_complete = False
        self.filePath = os.path.abspath(filePath)
        self.saveOnDestroy = saveOnDestroy
        if os.path.exists(self.filePath):
            self.data = pd.read_csv(self.filePath)
        else:
            self.data = self.create_base_data()
        self.init_complete = True

    def create_base_data(self):
        df = pd.DataFrame(columns=self.COLUMNS)
        return df

    def add_result(self,
                   result):
        self.data = pd.concat([self.data, result.to_pd_record()])

    def save(self):
        rootDir = os.path.dirname(self.filePath)
        if not os.path.exists(rootDir):
            os.makedirs(rootDir)
        self.data.to_csv(self.filePath, index=False)

    def __del__(self):
        if self.saveOnDestroy and self.init_complete:
            self.save()
