

# from xiwu.deploy.vicuna.vicuna_worker import run_worker
# from xiwu.deploy.xiwu.xiwu_worker import run_worker

from typing import Generator
import hepai
from xiwu.modules.deployer.worker_model import ModelArgs, WorkerArgs, WorkerModel, BaseWorkerModel
from xiwu.utils import general

if __name__ == '__main__':
    model_args, worker_args = hepai.parse_args((ModelArgs, WorkerArgs))  # 解析多个参数类
    model: BaseWorkerModel = WorkerModel(args=model_args)  # 可传入更多参数
    worker_args.description = worker_args.description or model.xmodel.get_description()
    worker_args.author = worker_args.author or model.xmodel.get_author()
    print(model_args)
    print(worker_args)
    if worker_args.test:
        general.test_model(model=model, stream=True)
    hepai.worker.start(model=model, worker_args=worker_args)




