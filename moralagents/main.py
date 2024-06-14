import os
from argparse import Namespace
from typing import List

import torch

from lm_steer.arguments import parse_args
from lm_steer.models.get_model import get_model
from moralagents.consts import SEED


def args_assert(args: Namespace):
    assert (
        args.prompt_only
        and args.steer_values is not None
        and args.ckpt_name is not None
    )


def generate(
    prompt_text: str,
    steer_values,
    tokenizer,
    model,
    prompt_length,
    num_beams,
    num_beam_groups,
    do_sample,
    temperature,
    top_p,
    device,
) -> str:
    token_length = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
    output = model.generate(
        prompt_text,
        steer_values,
        seed=SEED,
        max_length=token_length + prompt_length,
        min_length=token_length + prompt_length,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    output = output[len(prompt_text) :]
    return output


def main(args: Namespace):
    args_assert(args)

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model, tokenizer = get_model(
        args.model_name,
        args.adapted_component,
        args.adaptor_class,
        args.num_steers,
        args.rank,
        args.epsilon,
        args.init_var,
        args.low_resource_mode,
    )
    model.to_device(device)

    assert os.path.exists(args.ckpt_name)

    ckpt = torch.load(args.ckpt_name)
    model.load_state_dict(ckpt[1])

    model.eval()
    prompt_num = 25
    prompt_length = 20
    if args.eval_size is not None:
        prompt_data = prompt_data[: args.eval_size]
    num_beams = 1
    num_beam_groups = 1
    do_sample = True
    temperature = args.temperature
    steer_values: List[float] | None = (
        list(map(float, args.steer_values)) if args.steer_values is not None else None
    )

    def get_output(prompt_text: str, steer_values: List[float] | None) -> str:
        return generate(
            prompt_text,
            steer_values,
            tokenizer,
            model,
            prompt_length,
            num_beams,
            num_beam_groups,
            do_sample,
            temperature,
            args.top_p,
            device,
        )

    def get_full_output(prompt_text: str, steer_values: List[float] | None) -> str:
        output: str = get_output(prompt_text, steer_values)
        return f"{prompt_text}{output}"

    prompt_text: str = "I feel like "
    full_output: str = get_full_output(prompt_text, steer_values)

    print("===============")
    print(full_output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
