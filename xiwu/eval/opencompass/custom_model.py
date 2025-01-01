
# 新增API模型的评估


# from ..base_api import BaseAPIModel
from typing import Dict, List, Optional, Union
from opencompass.models.base_api import BaseAPIModel
from opencompass.models import OpenAI

# class MyModelAPI(BaseAPIModel):

#     is_api: bool = True

#     def __init__(self,
#                  path: str,
#                  max_seq_len: int = 2048,
#                  query_per_second: int = 1,
#                  retry: int = 2,
#                  meta_template: Optional[Dict] = None,
#                  **kwargs):
#         super().__init__(path=path,
#                          max_seq_len=max_seq_len,
#                          meta_template=meta_template,
#                          query_per_second=query_per_second,
#                          retry=retry)
#         ...

#     def generate(
#         self,
#         inputs,
#         max_out_len: int = 512,
#         temperature: float = 0.7,
#     ) -> List[str]:
#         """Generate results given a list of inputs."""
#         pass

#     def get_token_len(self, prompt: str) -> int:
#         """Get lengths of the tokenized string."""
#         pass

MY_API_BASE =  'http://localhost:8000/v1/chat/completions'

class MyModelAPI(OpenAI):

    is_api: bool = True

    def __init__(self,
                 path: str = 'gpt-3.5-turbo',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 key: Union[str, List[str]] = 'ENV',
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = None,
                 openai_api_base: str = MY_API_BASE,
                 openai_proxy_url: Optional[str] = None,
                 mode: str = 'none',
                 logprobs: Optional[bool] = False,
                 top_logprobs: Optional[int] = None,
                 temperature: Optional[float] = None,
                 tokenizer_path: Optional[str] = None,
                 extra_body: Optional[Dict] = None,
                 max_completion_tokens: int = 16384,
                 verbose: bool = False):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry,
                         key=key,
                         org=org,
                         meta_template=meta_template,
                         openai_api_base=openai_api_base,
                         openai_proxy_url=openai_proxy_url,
                         mode=mode,
                         logprobs=logprobs,
                         top_logprobs=top_logprobs,
                         temperature=temperature,
                         tokenizer_path=tokenizer_path,
                         extra_body=extra_body,
                         max_completion_tokens=max_completion_tokens,
                         verbose=verbose)
        
    def generate(self, inputs, max_out_len = 512, temperature = 0.7, **kwargs):
        return super().generate(inputs, max_out_len, temperature, **kwargs)
    
    def get_token_len(self, prompt: str) -> int:
        return super().get_token_len(prompt)