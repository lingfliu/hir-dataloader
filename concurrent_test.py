import multiprocessing

from concurrent_tools import MultiProcessPool
from array_tools import index_shuffle

def task_exclusive(a, b):
    return a+b

def task(a, b, queue):
    # print('task', a + b)
    queue.put(a + b)


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    queue = multiprocessing.Manager().Queue(maxsize=5000)
    for x in range(100):
        pool.apply_async(task, args=(x, x, queue))
    pool.close()
    pool.join()
    try:
        q = queue.get_nowait()
        while q is not None:
            print(q)
            q = queue.get_nowait()
    except:
        print('native pool done')

    pool = MultiProcessPool()
    array = index_shuffle(1000)
    for x in array:
        pool.submit('task', task_exclusive, (x, x))
    pool.subscribe()
    results = pool.fetch_results('task')
    for i, r in enumerate(results):
        print('result for 2*',  array[i], ' is ', r)
    pool.cleanup()
    print('custom pool done')

