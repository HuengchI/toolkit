import re
from tqdm import tqdm
import pandas as pd
from .multi_gpu_scheduler import MultiGPUTaskScheduler


class MultiGPUTaskSchedulerReRunner:
    def __init__(self, pipeline_execution_report_file, available_gpu_ids_str: str, device_arg_option: str=None, verbose=True) -> None:
        self.pipeline_execution_report_file = pipeline_execution_report_file
        self.available_gpu_ids = [int(s.strip())
                         for s in available_gpu_ids_str.split(',')]
        self.device_arg_option = device_arg_option
        self.verbose = verbose
        self.task_scheduler = MultiGPUTaskScheduler(available_gpu_ids=self.available_gpu_ids, realtime_subprocess_output=verbose)

        inputs = pd.read_csv(pipeline_execution_report_file)
        self.failed_inputs = inputs[inputs['success_flag']==False].reset_index(drop=True)

    def _ifprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def run(self, dry_run=False):

        self._ifprint(f'[In Progress] Availabel GPU IDs: {self.available_gpu_ids}')
        self._ifprint(f'[In Progress] Collected failed tasks: {len(self.failed_inputs)}')

        if '_rerun' in self.pipeline_execution_report_file:
            serial = int(re.findall(r'_rerun(\d+)', self.pipeline_execution_report_file)[0])
            serial += 1
            report_file_path = re.sub(r'_rerun(\d+)', f"_rerun{serial}", self.pipeline_execution_report_file)
        else:
            name = self.pipeline_execution_report_file.split('.')
            name.insert(1, '_rerun1.')
            report_file_path = ''.join(name)

        if not dry_run:
            # emit sampling tasks
            for _,row in self.failed_inputs.iterrows():
                cmd = eval(row['task_cmd'])
                if self.device_arg_option:
                    cmd = cmd[:cmd.index(f"--{self.device_arg_option}")] # strip old device_arg_option if any
                env = eval(row['task_env']) if 'task_env' in row else None
                self.task_scheduler.emit_task(task_cmd=cmd, process_env=env, device_arg_option=None, setting_device_env=True)

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

        self._ifprint(f'[Finished] Execution report saved to {report_file_path}')


if __name__=='__main__':
    import argparse
    from toolkit import MultiGPUTaskSchedulerReRunner
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_execution_report_file', type=str, required=True,
                        help='The execution report file generated by the Task Scheduler')
    parser.add_argument('--available_gpu_ids', type=str, required=True,
                    help='specifying multiple gpus for concurrent inference')
    parser.add_argument('--device_arg_option', type=str, default=None)

    args = parser.parse_args()

    re_runner = MultiGPUTaskSchedulerReRunner(args.pipeline_execution_report_file, args.available_gpu_ids, args.device_arg_option)

    re_runner.run()