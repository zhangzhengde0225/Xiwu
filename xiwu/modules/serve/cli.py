
from typing import Optional
import time
import json
import torch
import os

from xiwu.apis.xiwu_api import BaseArgs, XAssembler

from xiwu.apis.fastchat_api import (
    ExllamaConfig, XftConfig, GptqConfig, AWQConfig, 
    SimpleChatIO, RichChatIO, ProgrammaticChatIO, ChatIO,
    str_to_torch_dtype, get_generate_stream_function,
    get_context_length, get_conv_template, get_conversation_template,
    Conversation,
)
from xiwu import ASSEMBLER


class CLI:

    @classmethod
    def main(cls, args: BaseArgs):
        if args.gpus:
            if len(args.gpus.split(",")) < args.num_gpus:
                raise ValueError(
                    f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            os.environ["XPU_VISIBLE_DEVICES"] = args.gpus
        if args.enable_exllama:
            exllama_config = ExllamaConfig(
                max_seq_len=args.exllama_max_seq_len,
                gpu_split=args.exllama_gpu_split,
                cache_8bit=args.exllama_cache_8bit,
            )
        else:
            exllama_config = None
        if args.enable_xft:
            xft_config = XftConfig(
                max_seq_len=args.xft_max_seq_len,
                data_type=args.xft_dtype,
            )
            if args.device != "cpu":
                print("xFasterTransformer now is only support CPUs. Reset device to CPU")
                args.device = "cpu"
        else:
            xft_config = None
        if args.style == "simple":
            chatio = SimpleChatIO(args.multiline)
        elif args.style == "rich":
            chatio = RichChatIO(args.multiline, args.mouse)
        elif args.style == "programmatic":
            chatio = ProgrammaticChatIO()
        else:
            raise ValueError(f"Invalid style for console: {args.style}")
        try:
            CLI.chat_loop(
                args.model_path,
                args.device,
                args.num_gpus,
                args.max_gpu_memory,
                str_to_torch_dtype(args.dtype),
                args.load_8bit,
                args.cpu_offloading,
                args.conv_template,
                args.conv_system_msg,
                args.temperature,
                args.repetition_penalty,
                args.max_new_tokens,
                chatio,
                gptq_config=GptqConfig(
                    ckpt=args.gptq_ckpt or args.model_path,
                    wbits=args.gptq_wbits,
                    groupsize=args.gptq_groupsize,
                    act_order=args.gptq_act_order,
                ),
                awq_config=AWQConfig(
                    ckpt=args.awq_ckpt or args.model_path,
                    wbits=args.awq_wbits,
                    groupsize=args.awq_groupsize,
                ),
                exllama_config=exllama_config,
                xft_config=xft_config,
                revision=args.revision,
                judge_sent_end=args.judge_sent_end,
                debug=args.debug,
                history=not args.no_history,
            )
        except KeyboardInterrupt:
            print("exit...")

    @classmethod
    def chat_loop(cls,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        dtype: Optional[torch.dtype],
        load_8bit: bool,
        cpu_offloading: bool,
        conv_template: Optional[str],
        conv_system_msg: Optional[str],
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        chatio: ChatIO,
        gptq_config: Optional[GptqConfig] = None,
        awq_config: Optional[AWQConfig] = None,
        exllama_config: Optional[ExllamaConfig] = None,
        xft_config: Optional[XftConfig] = None,
        revision: str = "main",
        judge_sent_end: bool = True,
        debug: bool = True,
        history: bool = True,
    ):
        # Model
        model, tokenizer = ASSEMBLER.load_model(
            model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            gptq_config=gptq_config,
            awq_config=awq_config,
            exllama_config=exllama_config,
            xft_config=xft_config,
            revision=revision,
            debug=debug,
        )
        generate_stream_func = get_generate_stream_function(model, model_path)

        model_type = str(type(model)).lower()
        is_t5 = "t5" in model_type
        is_codet5p = "codet5p" in model_type
        is_xft = "xft" in model_type

        # Hardcode T5's default repetition penalty to be 1.2
        if is_t5 and repetition_penalty == 1.0:
            repetition_penalty = 1.2

        # Set context length
        context_len = get_context_length(model.config)

        # Chat
        def new_chat():
            if conv_template:
                conv = get_conv_template(conv_template)
            else:
                conv = ASSEMBLER.get_conversation_template(model_path)
            if conv_system_msg is not None:
                conv.set_system_message(conv_system_msg)
            return conv

        def reload_conv(conv: Conversation):
            """
            Reprints the conversation from the start.
            """
            for message in conv.messages[conv.offset :]:
                chatio.prompt_for_output(message[0])
                chatio.print_output(message[1])

        conv: Conversation = None

        while True:
            if not history or not conv:
                conv = new_chat()

            try:
                inp = chatio.prompt_for_input(conv.roles[0])
            except EOFError:
                inp = ""

            if inp == "!!exit" or not inp:
                print("exit...")
                break
            elif inp == "!!reset":
                print("resetting...")
                conv = new_chat()
                continue
            elif inp == "!!remove":
                print("removing last message...")
                if len(conv.messages) > conv.offset:
                    # Assistant
                    if conv.messages[-1][0] == conv.roles[1]:
                        conv.messages.pop()
                    # User
                    if conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()
                    reload_conv(conv)
                else:
                    print("No messages to remove.")
                continue
            elif inp == "!!regen":
                print("regenerating last message...")
                if len(conv.messages) > conv.offset:
                    # Assistant
                    if conv.messages[-1][0] == conv.roles[1]:
                        conv.messages.pop()
                    # User
                    if conv.messages[-1][0] == conv.roles[0]:
                        reload_conv(conv)
                        # Set inp to previous message
                        inp = conv.messages.pop()[1]
                    else:
                        # Shouldn't happen in normal circumstances
                        print("No user message to regenerate from.")
                        continue
                else:
                    print("No messages to regenerate.")
                    continue
            elif inp.startswith("!!save"):
                args = inp.split(" ", 1)

                if len(args) != 2:
                    print("usage: !!save <filename>")
                    continue
                else:
                    filename = args[1]

                # Add .json if extension not present
                if not "." in filename:
                    filename += ".json"

                print("saving...", filename)
                with open(filename, "w") as outfile:
                    json.dump(conv.dict(), outfile)
                continue
            elif inp.startswith("!!load"):
                args = inp.split(" ", 1)

                if len(args) != 2:
                    print("usage: !!load <filename>")
                    continue
                else:
                    filename = args[1]

                # Check if file exists and add .json if needed
                if not os.path.exists(filename):
                    if (not filename.endswith(".json")) and os.path.exists(
                        filename + ".json"
                    ):
                        filename += ".json"
                    else:
                        print("file not found:", filename)
                        continue

                print("loading...", filename)
                with open(filename, "r") as infile:
                    new_conv = json.load(infile)

                conv = get_conv_template(new_conv["template_name"])
                conv.set_system_message(new_conv["system_message"])
                conv.messages = new_conv["messages"]
                reload_conv(conv)
                continue

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            if is_codet5p:  # codet5p is a code completion model.
                prompt = inp

            gen_params = {
                "model": model_path,
                "prompt": prompt,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "max_new_tokens": max_new_tokens,
                "stop": conv.stop_str,
                "stop_token_ids": conv.stop_token_ids,
                "echo": False,
            }

            try:
                chatio.prompt_for_output(conv.roles[1])
                output_stream = generate_stream_func(
                    model,
                    tokenizer,
                    gen_params,
                    device,
                    context_len=context_len,
                    judge_sent_end=judge_sent_end,
                )
                t = time.time()
                outputs = chatio.stream_output(output_stream)
                duration = time.time() - t
                conv.update_last_message(outputs.strip())

                if debug:
                    num_tokens = len(tokenizer.encode(outputs))
                    msg = {
                        "conv_template": conv.name,
                        "prompt": prompt,
                        "outputs": outputs,
                        "speed (token/s)": round(num_tokens / duration, 2),
                    }
                    print(f"\n{msg}\n")

            except KeyboardInterrupt:
                print("stopped generation.")
                # If generation didn't finish
                if conv.messages[-1][1] is None:
                    conv.messages.pop()
                    # Remove last user message, so there isn't a double up
                    if conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()

                    reload_conv(conv)
