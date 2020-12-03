import pytorch_lightning as pl
import concurrent.futures

class Task:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        return self.func(*self.args, **self.kwargs)

class MultiTaskRunner:
    def __init__(self, gpus):
        self.gpus = gpus
        self.tasks = []

    def append_task(self, task):
        self.tasks.append(task)

    def run_all_tasks(self):