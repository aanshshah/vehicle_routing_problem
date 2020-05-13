from multiprocessing import Pool
import multiprocessing
def f(strategy):
    return strategy.run()

def run_processes(strategies):
    with Pool(multiprocessing.cpu_count()) as p:
        result = p.map(f,strategies)
    return result
