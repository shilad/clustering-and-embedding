import os.path
import shutil

import time


class ResultDir:
    def __init__(self):
        self.parent_dir = None
        for path in ('./results', '../results'):
            if os.path.isdir(path):
                self.parent_dir = path
                break
        else:
            raise Exception, 'No result directory found!'
        self.dir = None

    def get(self):
        if not self.dir:
            n = max([-1] +  [
                int(d)
                for d in os.listdir(self.parent_dir)
                if all(s.isdigit() for s in d)])
            self.dir = self.parent_dir + '/' + str(n+1)
            os.makedirs(self.dir)
            self.log('created on ' + time.ctime())

        return self.dir

    def log(self, message):
        with open(self.get() + '/log.txt', 'a') as f:
            f.write(message + '\n')


    def __str__(self):
        return self.get()

    def log_and_print(self, message):
        print(message)
        self.log(message)






