"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
from math import e
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
import webdataset as wds
from minigpt4.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import is_url
from minigpt4.datasets.data_utils import concat_datasets, reorg_datasets_by_split, ChainDataset
from minigpt4.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from minigpt4.runners.runner_base import RunnerBase
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

@registry.register_runner("rec_runner_base")
class RecRunnerBase(RunnerBase):
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    @torch.no_grad()
    def eval_epoch_pre(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        self.model.eval()
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model, data_loader, eval_text=self.eval_text)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )
    
    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model=model, data_loaders=data_loader, eval_text=self.eval_text)
        return results


    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False, tag=""):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()

        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        # save_obj = {
        #     "model": state_dict,
        #     "optimizer": self.optimizer.state_dict(),
        #     "config": self.config.to_dict(),
        #     "scaler": self.scaler.state_dict() if self.scaler else None,
        #     "epoch": cur_epoch,
        # }
        # save_to = os.path.join(
        #     self.output_dir,
        #     "checkpoint_{}{}.pth".format("best" if is_best else cur_epoch, tag),
        # )
        # logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        # torch.save(save_obj, save_to)
        for k in list(state_dict.keys()):
            if 'lora_' in k :
                del state_dict[k]
                save_lora = True
            elif 'base_model' in k:
                del state_dict[k]
        if len(list(state_dict.keys())) > 0:
            torch.save(state_dict, os.path.join(self.output_dir, f"project_model_{tag}.pth"))
        if save_lora:
            # 保存 LoRA 适配器参数
            os.makedirs(os.path.join(self.output_dir, f"adapter_{tag}"),exist_ok=True)
            try:
                lora_state_dict = get_peft_model_state_dict(model_no_ddp.llama_model_lora,adapter_name='group0')
                lora_config = model_no_ddp.llama_model_lora.peft_config['group0'] # 获取LoRA配置信息
            except:
                lora_state_dict = get_peft_model_state_dict(model_no_ddp.llama_model_lora)
                lora_config = model_no_ddp.llama_model_lora.peft_config['default'] # 获取LoRA配置信息
            torch.save(lora_state_dict, os.path.join(self.output_dir, f"adapter_{tag}", f"adapter_model.bin"))
            lora_config_file = os.path.join(self.output_dir, f"adapter_{tag}", f"adapter_config.json")
            if not os.path.exists(lora_config_file):
                adapter_config = {
                    'peft_type': "LORA",
                    'r': int(lora_config.r),
                    'lora_alpha': int(lora_config.lora_alpha),
                    'target_modules': list(lora_config.target_modules),
                    'lora_dropout': float(lora_config.lora_dropout),
                    'bias': str(lora_config.bias),
                    'task_type': str(lora_config.task_type),
                    'base_model_name_or_path': str(lora_config.base_model_name_or_path)
                }
                with open(os.path.join(self.output_dir, f"adapter_{tag}", f"adapter_config.json"),'w') as f:
                    json.dump(adapter_config,f,indent=4)
            # print(model_no_ddp.llama_model_lora.peft_config,type(model_no_ddp.llama_model_lora.peft_config))
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, self.output_dir))
            # model_no_ddp.llama_model_lora.save_pretrained(os.path.join(self.output_dir, f"adapter_{tag}"))