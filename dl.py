import os
cache = "/src/weights/"
os.environ["TORCH_HOME"] = "/src/weights/"
os.environ["HF_HOME"] = "/src/weights/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/src/weights/"
if not os.path.exists(cache):
    os.makedirs(cache)

import argparse
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat
import argparse


# Set environment variables for caches

parser = argparse.ArgumentParser(description="Demo")
parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
args = parser.parse_args(["--cfg-path","eval_configs/minigpt4_eval.yaml"])

cfg = Config(args)
print('Download Chat')

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

print('Loaded Model')
vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Download Finished')
