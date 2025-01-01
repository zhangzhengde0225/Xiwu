


from vllm import LLM

# from vllm

from hepai import HRModel

class CustomWorkerModel(HRModel):  # Define a custom worker model inheriting from HRModel.
    def __init__(self, name: str = "meta/llama3-8b", **kwargs):
        super().__init__(name=name, **kwargs)
        # lazy_load = kwargs.get("lazy_load", False)
        
        # model_dir="/aifs/user/data/zdzhang/models"
        # self.model_path = f'{model_dir}/llama3_8b_instruct'
        self.model_path = "meta-llama/Llama-3.1-8B-Instruct"
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLM(model=self.model_path, task="generate")
        return self._llm    

    
    @HRModel.remote_callable  # Decorate the function to enable remote call.
    def generate(self, prompt: str, **kwargs) -> str:
        """Define your custom method here."""
        from vllm import SamplingParams
        sampling_params = SamplingParams(**kwargs)
        rst = self.llm.generate(prompt, sampling_params=sampling_params)
        return rst

if __name__ == "__main__":


    CustomWorkerModel.run()  # Run the custom worker model.
    
