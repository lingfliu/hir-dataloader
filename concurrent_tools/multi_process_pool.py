import multiprocessing
from queue import Empty


def tuple_sort(list):
    return [x[1] for x in sorted(list, key=lambda x: x[0])]

def _tag_task(idx, task, params, queue):
    # print('task param=', params)
    result = task(*params)
    queue.put((idx, result))


class MultiProcessPool:
    def __init__(self, num_procs=4):
        cpu_count = multiprocessing.cpu_count()
        num_procs = num_procs if num_procs < cpu_count*0.8 else cpu_count*0.8
        self.num_procs = int(num_procs)
        self.task_pool = multiprocessing.Pool(processes=self.num_procs)
        self.queue = {}
        self.queue_idx = {}

    def submit(self, tag, task, params):
        if tag not in self.queue.keys():
            self.queue[tag] = multiprocessing.Manager().Queue(maxsize=5000)
            self.queue_idx[tag] = 0
        # _tag_task(tag, task, params, self.queue[tag])
        self.task_pool.apply_async(_tag_task, (self.queue_idx[tag], task, params, self.queue[tag]))
        self.queue_idx[tag] += 1

    def subscribe(self):

        self.task_pool.close()
        self.task_pool.join()
        return

    def fetch_results(self, tag):
        if tag in self.queue.keys():
            q = self.queue[tag]
            results = []
            try:
                r = q.get_nowait()
                while r is not None:
                    results.append(r)
                    r = q.get_nowait()
            except Empty:
                if len(results) > 0:
                    return tuple_sort(results)
                else:
                    return []
        else:
            return None

    def cleanup(self):
        self.queue.clear()