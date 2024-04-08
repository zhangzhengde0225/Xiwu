
import os, sys
from pathlib import Path
here = Path(__file__).parent
import argparse
try:
    from xiwu.apis.fastchat_api import add_model_args, main
except:
    sys.path.append(f'{here.parent.parent}')
    from xiwu.apis.fastchat_api import add_model_args, main
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    
    args.model_path = f'/data/zzd/weights/vicuna/vicuna-7b-v1.5-16k'
    args.model_path = f'/data/zzd/weights/vicuna/vicuna-7b'
    # args.num_gpus = 1  # 第2块GPU
    # args.gpus = "1"
    main(args)
    