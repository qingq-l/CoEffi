import io
import os
import time
import signal
import tempfile
import platform
import contextlib
import faulthandler
from tqdm import tqdm
from collections import defaultdict, Counter
from multiprocessing import Manager, Process
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

class TimeoutException(Exception):
    pass

class RedirectStdin(contextlib._RedirectStream):
    _stream = 'stdin'

class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs): raise IOError
    def readline(self, *args, **kwargs): raise IOError
    def readlines(self, *args, **kwargs): raise IOError
    def readable(self, *args, **kwargs): return False

class Sandbox(object):
    @staticmethod
    @contextlib.contextmanager
    def time_limit(seconds: float):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.signal(signal.SIGALRM, signal_handler)
        try: yield
        finally: signal.setitimer(signal.ITIMER_REAL, 0)

    @staticmethod
    @contextlib.contextmanager
    def swallow_io():
        stream = WriteOnlyStringIO()
        with contextlib.redirect_stdout(stream):
            with contextlib.redirect_stderr(stream):
                with RedirectStdin(stream): yield

    @staticmethod
    @contextlib.contextmanager
    def create_tempdir():
        with tempfile.TemporaryDirectory() as dirname:
            with Sandbox.chdir(dirname): yield dirname

    @staticmethod
    @contextlib.contextmanager
    def chdir(root):
        if root == ".":
            yield
            return
        cwd = os.getcwd()
        os.chdir(root)
        try: yield
        except BaseException as exc: raise exc
        finally: os.chdir(cwd)

    @staticmethod
    def reliability_guard(maximum_memory_bytes: Optional[int] = None):
        if maximum_memory_bytes is not None:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
            if platform.uname().system != 'Darwin':
                resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

        faulthandler.disable()
        import builtins
        builtins.exit = None
        builtins.quit = None

        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.kill = os.system = os.putenv = os.remove = os.removedirs = os.rmdir = os.fchdir = os.setuid = os.fork = os.forkpty = os.killpg = os.rename = os.renames = os.truncate = os.replace = os.unlink = os.fchmod = os.fchown = os.chmod = os.chown = os.chroot = os.fchdir = os.lchflags = os.lchmod = os.lchown = os.getcwd = os.chdir = None

        import shutil
        shutil.rmtree = shutil.move = shutil.chown = None

        import subprocess
        subprocess.Popen = None

        __builtins__['help'] = None

        import sys
        sys.modules['ipdb'] = sys.modules['joblib'] = sys.modules['resource'] = sys.modules['psutil'] = sys.modules['tkinter'] = None

    @staticmethod
    def unsafe_execute(sample, result):
        runtime = 0
        import traceback

        with Sandbox.create_tempdir():
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            Sandbox.reliability_guard()

            try:
                namespace = {}
                exec("import sys", namespace)
                exec("sys.path.insert(0, '/root/autodl-tmp/SkelDPO/Mercury/Mercury-main/src')", namespace)

                exec("import re, itertools, collections, heapq, bisect, string, functools, math, copy", namespace)
                try: exec("import lctk", namespace)
                except: pass
                try: exec("import sortedcontainers", namespace)
                except: pass

                exec("from math import floor, ceil, factorial, sqrt, inf", namespace)
                exec("from sys import maxsize, stdin", namespace)
                exec("from bisect import bisect_left, bisect_right", namespace)
                exec("from itertools import permutations, zip_longest", namespace)
                exec("from heapq import heappush, heappop, heapify", namespace)
                exec("from collections import deque, defaultdict, OrderedDict", namespace)
                exec("from typing import List, Optional, Tuple", namespace)
                exec("from functools import lru_cache, cache", namespace)

                exec("class ListNode(object):\n\tdef __init__(self, val=0, next=None):\n\t\tself.val = val\n\t\tself.next = next", namespace)
                exec("class TreeNode(object):\n\tdef __init__(self, val=0, left=None, right=None):\n\t\tself.val = val\n\t\tself.left = left\n\t\tself.right = right", namespace)

                exec("def print(*args):pass", namespace)

                total, passed = 0, 0
                with Sandbox.swallow_io():
                    with Sandbox.time_limit(sample['timeout']):
                        try:
                            code = sample['solution']
                            import re
                            class_matches = list(re.finditer(r'^class\s+', code, flags=re.MULTILINE))
                            if len(class_matches) > 1:
                                code = code[class_matches[-1].start():]

                            exec(code, namespace)
                            exec(f"solution=Solution()", namespace)
                            exec(sample['convert_offline'], namespace)
                            exec(sample['evaluate_offline'], namespace)
                        except Exception as e:
                            result.append({"status": f"failed@load: {traceback.format_exc()}", "runtime": runtime})
                            return

                        try:
                            start_time = time.time()
                            for test_case in sample['test_cases']:
                                namespace['inputs'] = test_case['input']
                                namespace['expected'] = test_case['expected']
                                exec("inputs, expected = convert_offline((inputs, expected))", namespace)
                                exec(f"outputs = solution.{sample['entry_point']}(*inputs)", namespace)
                                exec(f"passed = evaluate_offline(inputs, outputs, expected)", namespace)
                                total += 1
                                passed += (1 if namespace['passed'] else 0)
                            end_time = time.time()
                            runtime = end_time-start_time
                        except Exception as e:
                            result.append({"status": f"failed@eval: {traceback.format_exc()}", "runtime": runtime})
                            return

                if total == passed:
                    result.append({"status": "passed", "runtime": runtime})
                else:
                    result.append({"status": "failed@cases", "runtime": runtime})

            except TimeoutException:
                result.append({"status": "timeout", "runtime": runtime})
            except BaseException as e:
                result.append({"status": f"failed@error: {traceback.format_exc()}", "runtime": runtime})
            finally:
                shutil.rmtree = rmtree
                os.rmdir = rmdir
                os.chdir = chdir

    @staticmethod
    def run_sample(sample) -> Dict:
        with Manager() as manager:
            result = manager.list()
            p = Process(target=Sandbox.unsafe_execute, args=(sample, result))
            p.start()
            p.join(timeout=sample['timeout']+1)
            if p.is_alive(): p.kill()
            if not result: result.append({"status": "timeout", "runtime": sample['timeout']})
            return dict(result=result[0]['status'], runtime=result[0]['runtime'], index=sample['solution_index'])

    @staticmethod
    def run_samples(samples, n_workers=4):
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures, results = list(), list()
            for sample in samples:
                args = (sample,)
                future = executor.submit(Sandbox.run_sample, *args)
                futures.append(future)
            for future in tqdm(as_completed(futures), total=len(futures), desc='Evaluating'):
                results.append(future.result())
        return results