import os
import time
import subprocess
from typing import List, Tuple, Union
from multiprocessing import Manager, Process
import pandas as pd


def run_subprocess_with_retry(cmd, env=None, max_retries=3, realtime_output=True):
    retry = 0

    last_try_output = None
    while retry < max_retries:
        try:
            if not realtime_output:
                process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                output = stdout.decode('utf-8').strip(), stderr.decode('utf-8').strip()
            else:
                process = subprocess.Popen(cmd, env=env)
                output = process.communicate()
                # output = '', ''
            last_try_output = output
        except Exception as e:
            return False, str(cmd), f'subprocess.Popen Error {e}'

        if process.returncode == 0:
            return True, *output
        else:
            print(f"[Failed] {' '.join(cmd)}")
            print(f"[Stdout] {output[0]}")
            print(f"[Stderr] {output[1]}")
            print(
                f"Process failed, retrying... (retry {retry+1}/{max_retries})")
            process.terminate()
            retry += 1
            delay = 2**retry  # exponential backoff
            print(f"Waiting for {delay} seconds before retrying...")
            time.sleep(delay)
    print(f"[All Failed] Exceeded maximum retries. Exiting.")

    return False, *last_try_output

class MultiGPUTaskScheduler:
    def __init__(self, available_gpu_ids: List[int], realtime_subprocess_output=False) -> None:
        self.manager = Manager()
        self.task_queue = self.manager.Queue()
        self.progress_queue = self.manager.Queue()
        self.realtime_subprocess_output = realtime_subprocess_output

        # Launch sub-processes and let them wait
        self.processes = []
        for gpu_id in available_gpu_ids:
            process = Process(target=MultiGPUTaskScheduler.gpu_proc_worker, args=(
                gpu_id, self.task_queue, self.progress_queue, self.realtime_subprocess_output))
            self.processes.append(process)
            process.start()
        
        self.total_task_cnt = 0
        self.execution_feed_back = []

    def emit_task(self, task_cmd: List, process_env: dict, device_arg_option:Union[str, None]=None, setting_device_env: bool=False) -> None:
        """Emit a task to the process pool.

        Args:
            task_cmd (List): A list representing the command line arguments for the task to be executed.
            process_env (dict): The environment variables of the task.
            device_arg_option (Union[str, None], optional): If provided, the GPU ID will be appended to the task_cmd 
                list in the form of "--{device_arg_option} {gpu_id}". Defaults to None.
            setting_device_env (bool, optional): Whether to set the GPU ID in the task environment (CUDA_VISIBLE_DEVICES). 
                Defaults to False.
        """
        self.task_queue.put((task_cmd, dict(process_env), device_arg_option, setting_device_env)) ##emit task to subprocesses
        self.total_task_cnt+=1

    def notify_exit(self, wait_exiting: bool = True) -> None:
        """Notify all processes to exit and stop emitting any more tasks.

        Args:
            wait_exiting (bool, optional): Indicates whether the function should block and wait for all processes to exit. 
            Defaults to True.
        """
        # Inform each process to terminate and wait everything finishes
        for process in self.processes:
            self.task_queue.put(None)
        # Wait all processes exit
        if wait_exiting:
            for process in self.processes:
                process.join()
            print('[system] All Sub-processes Exited')

    def pull_process_message(self) -> Tuple[List, dict, bool, str, str]:
        """Retrieve messages from processes, if any. If there are currently no messages, this function will block and wait.

        Returns:
            Tuple[List, dict, bool, str, str]: A message from a process, which is a tuple composed of:
                - The executed task command (task_cmd)
                - The environment variables of the executed task (task_env)
                - The execution success status (True for successful, False for failed)
                - The standard output (stdout) of the task execution
                - The standard error (stderr) of the task execution
        """
        self.execution_feed_back.append(self.progress_queue.get())
        return self.execution_feed_back[-1]

    def report_progress(self) -> pd.DataFrame:
        """Get a execution report of all finished tasks by now

        Returns:
            pandas.DataFrame: the report in format of a pandas dataframe
        """
        execution_feed_back = pd.DataFrame(self.execution_feed_back)
        execution_feed_back.columns = ['task_cmd', 'task_env', 'success_flag', 'process_stdout', 'process_stderr']
        return execution_feed_back[['task_cmd', 'success_flag', 'process_stdout', 'process_stderr', 'task_env']]

    @classmethod
    def gpu_proc_worker(cls, gpu_id: int, task_queue, progress_queue, realtime_subprocess_output: bool):
        for task_spec in iter(task_queue.get, None):  # Terminate when task is 'None'
            task_cmd, process_env, device_arg_option, setting_device_env = task_spec
            if device_arg_option is not None:
                task_cmd.extend([f'--{device_arg_option}', str(gpu_id)])
            if setting_device_env:
                process_env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            success_flag, process_stdout, process_stderr = run_subprocess_with_retry(
                cmd=task_cmd,
                env=process_env,
                max_retries=3,
                realtime_output=realtime_subprocess_output
            )

            progress_queue.put((
                task_cmd, process_env, success_flag, process_stdout, process_stderr,
            ))  # Inform main process
        print(f'[Sub-process] Process {os.getpid()} exit successfully')
        return

if __name__ == '__main__':
    import os
    from tqdm import tqdm
    task_scheduler = MultiGPUTaskScheduler(available_gpu_ids=[1, 2, 3, 4])

    # emit tasks
    for i in range(40):
        if i % 2:
            task_cmd = ['bash -c', str(i)] # expect to fail
        else:
            task_cmd = ['echo', str(i)]

        task_scheduler.emit_task(task_cmd=task_cmd, process_env=os.environ, device_arg_option='DEVICE', setting_device_env=True)

    # Listen process execution progress
    pbar = tqdm(total=task_scheduler.total_task_cnt)
    for _ in range(task_scheduler.total_task_cnt):
        task_scheduler.pull_process_message()
        pbar.update()
    pbar.close()

    # No more tasks and inform to terminate
    task_scheduler.notify_exit(wait_exiting=True)

    # Get execution reports
    file_path = 'pipeline_execution_report.csv'
    task_scheduler.report_progress().to_csv(file_path, index=False)
    print(f'[Finished] Execution report saved to {file_path}')