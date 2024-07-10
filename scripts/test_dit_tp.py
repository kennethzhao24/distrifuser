import argparse
import os
import time

import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from tqdm import trange

from distrifuser.dit_pipeline import DistriDiTPipeline
from distrifuser.utils import DistriConfig


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline", type=str, default="dit", choices=["sdxl", "dit", "pixartalpha"]
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="generation",
        choices=["generation", "benchmark"],
        help="Purpose of running the script",
    )
    # Diffuser specific arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    )
    parser.add_argument("--labels", type=str, nargs="+", default=["panda"])
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs="*",
        default=1024,
        help="Image size of generation",
    )
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="dpm-solver",
        choices=["euler", "dpm-solver", "ddim"],
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # pipefuser specific arguments
    parser.add_argument(
        "--no_split_batch",
        action="store_true",
        help="Disable the batch splitting for classifier-free guidance",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=4, help="Number of warmup steps"
    )
    parser.add_argument(
        "--sync_mode",
        type=str,
        default="corrected_async_gn",
        choices=[
            "separate_gn",
            "async_gn",
            "corrected_async_gn",
            "sync_gn",
            "full_sync",
            "no_sync",
        ],
        help="Different GroupNorm synchronization modes",
    )
    parser.add_argument(
        "--parallelism",
        type=str,
        default="patch",
        choices=["patch", "tensor", "naive_patch", "pipefusion", "sequence"],
    )
    parser.add_argument(
        "--no_cuda_graph", action="store_true", help="Disable CUDA graph"
    )
    parser.add_argument(
        "--split_scheme",
        type=str,
        default="alternate",
        choices=["row", "col", "alternate"],
        help="Split scheme for naive patch",
    )

    # Benchmark specific arguments
    parser.add_argument(
        "--output_type", type=str, default="pil", choices=["latent", "pil"]
    )
    parser.add_argument(
        "--warmup_times", type=int, default=5, help="Number of warmup times"
    )
    parser.add_argument(
        "--test_times", type=int, default=20, help="Number of test times"
    )
    parser.add_argument(
        "--ignore_ratio",
        type=float,
        default=0.2,
        help="Ignored ratio of the slowest and fastest steps",
    )

    parser.add_argument("--pp_num_patch", type=int, default=2)
    parser.add_argument(
        "--no_use_resolution_binning",
        action="store_true",
    )
    args = parser.parse_args()
    return args


def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    args = get_args()

    if isinstance(args.image_size, int):
        args.image_size = [args.image_size, args.image_size]
    else:
        if len(args.image_size) == 1:
            args.image_size = [args.image_size[0], args.image_size[0]]
        else:
            assert len(args.image_size) == 2
    distri_config = DistriConfig(
        height=args.image_size[0],
        width=args.image_size[1],
        do_classifier_free_guidance=args.guidance_scale > 1,
        split_batch=not args.no_split_batch,
        warmup_steps=args.warmup_steps,
        mode=args.sync_mode,
        use_cuda_graph=not args.no_cuda_graph,
        parallelism=args.parallelism,
        split_scheme=args.split_scheme,
    )

    if args.model_path is None:
        if args.pipeline == "dit":
            args.model_path = "facebook/DiT-XL-2-256"
        elif args.pipeline == "sdxl":
            args.model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        elif args.pipeline == "pixartalpha":
            args.model_path = "PixArt-alpha/PixArt-XL-2-1024-MS"

    if args.parallelism != "pipefusion":
        if args.scheduler == "euler":
            scheduler = EulerDiscreteScheduler.from_pretrained(
                args.model_path, subfolder="scheduler"
            )
        elif args.scheduler == "dpm-solver":
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                args.model_path, subfolder="scheduler"
            )
        elif args.scheduler == "ddim":
            scheduler = DDIMScheduler.from_pretrained(
                args.model_path, subfolder="scheduler"
            )
        else:
            raise NotImplementedError

    if args.pipeline == "dit":
        pipeline = DistriDiTPipeline.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            distri_config=distri_config,
            # variant="fp16",
            # use_safetensors=True,
            scheduler=scheduler,
        )
        prompt = args.labels

    if args.mode == "generation":
        assert args.output_path is not None
        pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
        image = pipeline(
            prompt,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).images[0]
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        image.save(args.output_path)
    elif args.mode == "benchmark":
        pipeline.set_progress_bar_config(
            position=1, desc="Generation", leave=False, disable=distri_config.rank != 0
        )
        for i in trange(
            args.warmup_times,
            position=0,
            desc="Warmup",
            leave=False,
            disable=distri_config.rank != 0,
        ):
            pipeline(
                prompt,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                output_type=args.output_type,
            )
            torch.cuda.synchronize()
        latency_list = []
        for i in trange(
            args.test_times,
            position=0,
            desc="Test",
            leave=False,
            disable=distri_config.rank != 0,
        ):
            start_time = time.time()
            pipeline(
                prompt,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                output_type=args.output_type,
            )
            torch.cuda.synchronize()
            end_time = time.time()
            latency_list.append(end_time - start_time)
        latency_list = sorted(latency_list)
        ignored_count = int(args.ignore_ratio * len(latency_list) / 2)
        if ignored_count > 0:
            latency_list = latency_list[ignored_count:-ignored_count]
        if distri_config.rank == 0:
            memory = torch.cuda.max_memory_allocated(device="cuda")
            if args.output_path is not None:
                with open(f"{args.output_path}", "a") as f:
                    f.write(f"Info : {args} {distri_config.__dict__}\n")
                    f.write(f"Latency: {sum(latency_list) / len(latency_list):.5f} s\n")
                    f.write(f"{latency_list}\n")
                    f.write(f"{memory / (1024**3)} GB\n")
            else:
                print(f"Info : {args} {distri_config.__dict__}")
                print(f"Latency: {sum(latency_list) / len(latency_list):.5f} s")
                print(f"{latency_list}")
                print(f"{memory / (1024**3)} GB\n")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
