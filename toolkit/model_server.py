import os
import time
from multiprocessing.connection import Listener, Client
from multiprocessing import process
from multiprocessing import Process
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from copy import deepcopy


class Hook:
    server_address_base: str
    self_gpu_id: int

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        start_time = time.time()

        # import dynamic modules
        cls.__name__ = "AutoModelForCausalLM"
        code_revision = kwargs.pop("code_revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, code_revision=code_revision, trust_remote_code=trust_remote_code)
        class_ref = config.auto_map[cls.__name__]
        _ = get_class_from_dynamic_module(
            class_ref, pretrained_model_name_or_path, code_revision=code_revision, **kwargs
        )

        # retrieve the shared model from server
        model_name = os.path.abspath(pretrained_model_name_or_path).rsplit(os.sep, maxsplit=1)[-1]
        address = os.path.join(cls.server_address_base, model_name)

        with Client(address, family='AF_UNIX') as conn:
            # due to the implementation details of pytorch.multiprocessing, shared tensors need server's authkey to be correctly unpickled
            original_auth_key = process.current_process().authkey

            auth_key = process.AuthenticationString(conn.recv())

            process.current_process().authkey = auth_key
            model = conn.recv()
            process.current_process().authkey = original_auth_key

        print(f"Shared model loaded from GPU:{model.real_gpu_id} within {time.time() - start_time:.2f}")
        return model


_original_from_pretrained = AutoModelForCausalLM.from_pretrained

def hook(self_gpu_id = None, server_address_base=None):
    # Hook
    AutoModelForCausalLM.from_pretrained = Hook.from_pretrained

    if not self_gpu_id:
        try:
            self_gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES'])
        except Exception as e:
            print(e)
            print(f"[Fatal] Failed to infer self_gpu_id, specify it explicitly")
    Hook.self_gpu_id = self_gpu_id

    if not server_address_base:
        server_address_base = "/tmp/model_sever"
    Hook.server_address_base = os.path.join(server_address_base, str(self_gpu_id))


class SharedModelServer:
    def __init__(self, model, identifying_name, real_gpu_id) -> None:
        """All arguments will be detour to transformers.AutoModel.from_pretrained()
        """
        self.model = model
        self.model.real_gpu_id = real_gpu_id
        self.model.share_memory()

        self.identifying_name = identifying_name
        self.real_gpu_id = real_gpu_id

    def _clear(self, address):
        # remove potentially previously existing socket files
        try:
            os.unlink(address)
        except OSError:
            if os.path.exists(address):
                raise

    def main_loop(self, server_address_base, backlog=20):
        os.makedirs(os.path.join(server_address_base, str(self.real_gpu_id)), exist_ok=True)
    
        address = os.path.join(server_address_base, str(self.real_gpu_id), self.identifying_name)
        self._clear(address)

        with Listener(address=address, family='AF_UNIX', backlog=backlog) as listener:
            print(f"Model server ({self.identifying_name}) started on GPU:{self.real_gpu_id}")
            try:
                while True:
                    with listener.accept() as conn:
                        print('New connection accepted.')
                        conn.send(bytes(process.current_process().authkey))  # send authkey
                        conn.send(self.model)
                        conn.close()
            except KeyboardInterrupt:
                print("Ctrl + C pressed, exit now")
            except Exception as e:
                print(e)
            finally:
                self._clear(address)

class MultiGPUSharedModelServerLauncher:
    def __init__(self, target_gpu_ids: str, server_address_base="/tmp/model_sever") -> None:
        self.target_gpu_ids = [int(s.strip())
                    for s in target_gpu_ids.split(',')]
        self.server_address_base = server_address_base

        self.processes = []

    # running on new process
    def model_dispatcher(self, gpu_id, pretrained_model_name_or_path):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path,
                                                    trust_remote_code = True,
                                                    torch_dtype=torch.bfloat16).to('cuda')

        model_name = os.path.abspath(pretrained_model_name_or_path).rsplit(os.sep, maxsplit=1)[-1]
 
        model_server = SharedModelServer(model=model,
                                            identifying_name=model_name,
                                            real_gpu_id=gpu_id)
        model_server.main_loop(self.server_address_base)

    def launch(self, pretrained_model_name_or_path):
        # Create servers executing on sub-processes
        for gpu_id in self.target_gpu_ids:
            process = Process(target=self.model_dispatcher, args=(gpu_id, pretrained_model_name_or_path))
            self.processes.append(process)

        for process in self.processes:
            process.start()

    def join(self):
        try:
            for process in self.processes:
                process.join()
        except KeyboardInterrupt:
            print("Ctrl + C pressed, exit now")
        except Exception as e:
            print(e)
        finally:
            self.kill()

    def kill(self):
        for process in self.processes:
            process.kill()
        print(f"All servers killed.")



if __name__ == "__main__":
    import argparse
    import torch
    from transformers import AutoModelForCausalLM
    from toolkit.model_server import MultiGPUSharedModelServerLauncher, hook

    args = argparse.Namespace
    args.base_model_path = "//data/huengchi/Research/federated/pretrain/nlp/Qwen-1_8B/"

    model_server = MultiGPUSharedModelServerLauncher(target_gpu_ids="1, 2")
    model_server.launch(args.base_model_path)

    time.sleep(30)

    hook(self_gpu_id=2)
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path,
                                                trust_remote_code = True,
                                                torch_dtype=torch.bfloat16).to('cuda')

    hook(self_gpu_id=1)
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path,
                                                trust_remote_code = True,
                                                torch_dtype=torch.bfloat16).to('cuda')

    model_server.join()
