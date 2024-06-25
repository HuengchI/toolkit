import subprocess
import time

def run_subprocess_with_retry(cmd, env=None, max_retries=3, realtime_output=False, delay_time_base = 2):
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
            delay = delay_time_base**retry  # exponential backoff
            print(f"Waiting for {delay} seconds before retrying...")
            time.sleep(delay)
    print(f"[All Failed] Exceeded maximum retries. Exiting.")

    return False, *last_try_output