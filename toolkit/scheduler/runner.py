from typing import List, Dict
from tqdm import tqdm
from .multi_gpu_scheduler import MultiGPUTaskScheduler


class MultiGPUTaskSchedulerRunner:
    def __init__(self, output_name, all_task_cmds: List[List], all_task_envs: List[Dict], available_gpu_ids_str: str, device_arg_option: str=None, setting_device_env=True, verbose=True) -> None:
        self.output_name = output_name
        self.all_task_cmds = all_task_cmds
        self.all_task_envs = all_task_envs
        self.available_gpu_ids = [int(s.strip())
                         for s in available_gpu_ids_str.split(',')]
        self.device_arg_option = device_arg_option
        self.setting_device_env = setting_device_env
        self.verbose = verbose
        self.task_scheduler = MultiGPUTaskScheduler(available_gpu_ids=self.available_gpu_ids, realtime_subprocess_output=verbose)

    def _ifprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def run(self, dry_run=False):
        print(f'[In Progress] Availabel GPU IDs: {self.available_gpu_ids}')

        report_file_path = f'pipeline_execution_report_{self.output_name}.csv'

        if not dry_run:
            # emit sampling tasks
            for task, env in zip(self.all_task_cmds, self.all_task_envs):
                self.task_scheduler.emit_task(task_cmd=task, process_env=env, device_arg_option=self.device_arg_option, setting_device_env=self.setting_device_env)

            # Listen process execution progress
            pbar = tqdm(total=self.task_scheduler.total_task_cnt, disable=not self.verbose)
            for _ in range(self.task_scheduler.total_task_cnt):
                self.task_scheduler.pull_process_message()
                pbar.update()
                self.task_scheduler.report_progress().to_csv(report_file_path, index=False) # Progressively execution report
            pbar.close()

            # Get final execution reports
            self.task_scheduler.report_progress().to_csv(report_file_path, index=False)

        # No more tasks and inform to terminate
        self.task_scheduler.notify_exit(wait_exiting=True)

        print(f'[Finished] Execution report saved to {report_file_path}')


if __name__=='__main__':
    import os
    from toolkit import MultiGPUTaskSchedulerRunner
    tasks = []
    envs = []
    # emit tasks
    for i in range(40):
        if i % 2:
            task_cmd = ['bash -c', str(i)] # expect to fail
        else:
            task_cmd = ['echo', str(i)]

        tasks.append(task_cmd)
        envs.append(os.environ)


    runner = MultiGPUTaskSchedulerRunner(output_name='test',
                                            all_task_cmds=tasks,
                                            all_task_envs=envs,
                                            available_gpu_ids_str="0, 1, 2, 3",
                                            device_arg_option="DeviceNum"
                                            )

    runner.run()
