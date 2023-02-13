import multiprocessing


def task(id, result_map):
    result_map[id] = id
    print('task', id, 'done')



if __name__ == '__main__':
    task_pool = multiprocessing.Pool(processes=14)
    mgr = multiprocessing.Manager()
    result_map = mgr.dict()

    for i in range(100):
        task_pool.apply_async(task, args=(i, result_map))

    task_pool.close()
    task_pool.join()

    for key in result_map:
        print(key, result_map[key])