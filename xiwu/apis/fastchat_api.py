import os, sys
from pathlib import Path
here = Path(__file__).parent



try:
    from fastchat import __version__
    flag = True
except:
    flag = False
    # fastchat_path = f'{here.parent}/repos/FastChat'
    # # 检查文件夹是否为空，如果是则提示下载方法
    # sys.path.insert(1, fastchat_path)
    # from fastchat import __version__
    # print(f'fastchat not installed, use local in `{fastchat_path}` with version {__version__}')
    

if flag:
    from fastchat.train.llama2_flash_attn_monkey_patch import (
        replace_llama_attn_with_flash_attn,
    )


    from fastchat.serve.inference import (
        load_model, generate_stream, get_conv_template, get_conversation_template,
        get_context_length, get_generate_stream_function, ChatIO
        )

    from fastchat.conversation import (
        conv_templates, Conversation, SeparatorStyle,
        register_conv_template,   
    )
    from fastchat.serve.cli import (
        SimpleChatIO, RichChatIO, ProgrammaticChatIO,
        add_model_args,
        main,
    )


    from fastchat.llm_judge.gen_api_answer import(
        get_answer, reorg_answer_file,
    )

    from fastchat.llm_judge.common import (
        load_questions,
        temperature_config,
        # chat_compeletion_openai,
        # chat_compeletion_anthropic,
        # chat_compeletion_palm,
        API_ERROR_OUTPUT,
        API_MAX_RETRY,
        API_RETRY_SLEEP,
        load_model_answers,
        load_judge_prompts,
        check_data,
        play_a_match_pair,
        play_a_match_single,
        get_model_list,
        Judge,
        MatchPair,
        MatchSingle,
        NEED_REF_CATS,
    )

    from fastchat.llm_judge.gen_judgment import (
        make_match_single,
        make_judge_single,
        make_judge_pairwise,
        make_match_all_pairs,
        make_match,
    )

    from fastchat.model.model_adapter import (
        get_conversation_template,
        BaseModelAdapter,
        register_model_adapter,
        model_adapters,
        raise_warning_for_incompatible_cpu_offloading_configuration,
    )

    from fastchat.modules.awq import AWQConfig, load_awq_quantized
    from fastchat.modules.gptq import GptqConfig, load_gptq_quantized
    from fastchat.modules.exllama import ExllamaConfig, load_exllama_model
    from fastchat.modules.xfastertransformer import XftConfig, load_xft_model


    from fastchat.utils import (
        get_gpu_memory,
        oai_moderation,
        moderation_filter,
        clean_flant5_ckpt,
        str_to_torch_dtype,
    )

    from fastchat.model.monkey_patch_non_inplace import (
        replace_llama_attn_with_non_inplace_operations
    )

    from fastchat.train.train import (
        make_supervised_data_module,
        trainer_save_model_safe,
        LazySupervisedDataset,
        SupervisedDataset,
        rank0_print,
        preprocess,
    )

    from fastchat.model.model_chatglm import generate_stream_chatglm
    from fastchat.model.model_codet5p import generate_stream_codet5p
    from fastchat.model.model_falcon import generate_stream_falcon
    from fastchat.model.model_yuan2 import generate_stream_yuan2
    from fastchat.model.model_exllama import generate_stream_exllama
    from fastchat.model.model_xfastertransformer import generate_stream_xft


else:
    from functools import partial
    def func(name: str):
        raise NotImplementedError(f'Function `{name}` is not imported from FastChat')
    
    class Object:

        def __init__(self, *args, **kwargs):
            raise NotImplementedError('Object is not imported from FastChat')
    
    
    get_generate_stream_function = partial(func, 'get_generate_stream_function')
    load_model = partial(func, 'load_model')
    get_conv_template = partial(func, 'get_conv_template')
    get_conversation_template = partial(func, 'get_conversation_template')
    get_context_length = partial(func, 'get_context_length')
    
    Conversation = Object
    GptqConfig, AWQConfig, ExllamaConfig, XftConfig =Object, Object, Object, Object
    BaseModelAdapter = Object