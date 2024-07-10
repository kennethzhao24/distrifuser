import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from diffusers.models.attention import FeedForward, GELU
from diffusers.models.attention_processor import Attention

from distrifuser.modules.base_module import BaseModule
from distrifuser.utils import DistriConfig


class AttnTP(BaseModule):
    def __init__(self, 
                 module: Attention, 
                 distri_config: DistriConfig):
        super(AttnTP, self).__init__(module, distri_config)

        heads = module.heads
        sliced_heads = heads // distri_config.n_device_per_batch # divide attention heads by number of devices
        remainder_heads = heads % distri_config.n_device_per_batch
        if distri_config.split_idx() < remainder_heads:
            sliced_heads += 1
        self.sliced_heads = sliced_heads

        if sliced_heads > 0:
            if distri_config.split_idx() < remainder_heads:
                start_head = distri_config.split_idx() * sliced_heads
            else:
                start_head = (
                    remainder_heads * (sliced_heads + 1) + (distri_config.split_idx() - remainder_heads) * sliced_heads
                )
            end_head = start_head + sliced_heads

            dim = module.to_q.out_features // heads  # get head_dim

            # sharded q_proj
            sharded_to_q = nn.Linear(
                module.to_q.in_features,
                sliced_heads * dim,
                bias=module.to_q.bias is not None,
                device=module.to_q.weight.device,
                dtype=module.to_q.weight.dtype,
            )
            sharded_to_q.weight.data.copy_(module.to_q.weight.data[start_head * dim : end_head * dim]) # copy weight
            if module.to_q.bias is not None:
                sharded_to_q.bias.data.copy_(module.to_q.bias.data[start_head * dim : end_head * dim])

            # sharded k_proj
            sharded_to_k = nn.Linear(
                module.to_k.in_features,
                sliced_heads * dim,
                bias=module.to_k.bias is not None,
                device=module.to_k.weight.device,
                dtype=module.to_k.weight.dtype,
            )
            sharded_to_k.weight.data.copy_(module.to_k.weight.data[start_head * dim : end_head * dim])
            if module.to_k.bias is not None:
                sharded_to_k.bias.data.copy_(module.to_k.bias.data[start_head * dim : end_head * dim])

            # sharded v_proj
            sharded_to_v = nn.Linear(
                module.to_v.in_features,
                sliced_heads * dim,
                bias=module.to_v.bias is not None,
                device=module.to_v.weight.device,
                dtype=module.to_v.weight.dtype,
            )
            sharded_to_v.weight.data.copy_(module.to_v.weight.data[start_head * dim : end_head * dim])
            if module.to_v.bias is not None:
                sharded_to_v.bias.data.copy_(module.to_v.bias.data[start_head * dim : end_head * dim])
            
            # sharded out_proj
            sharded_to_out = nn.Linear(
                sliced_heads * dim,
                module.to_out[0].out_features,
                bias=module.to_out[0].bias is not None,
                device=module.to_out[0].weight.device,
                dtype=module.to_out[0].weight.dtype,
            )
            sharded_to_out.weight.data.copy_(module.to_out[0].weight.data[:, start_head * dim : end_head * dim])
            if module.to_out[0].bias is not None:
                sharded_to_out.bias.data.copy_(module.to_out[0].bias.data)
            
            # delete original module
            del module.to_q
            del module.to_k
            del module.to_v

            old_to_out = module.to_out[0]

            # replace original modul with sharded module
            module.to_q = sharded_to_q
            module.to_k = sharded_to_k
            module.to_v = sharded_to_v
            module.to_out[0] = sharded_to_out
            module.heads = sliced_heads

            del old_to_out

            torch.cuda.empty_cache()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None):

        distri_config = self.distri_config
        module = self.module
        residual = hidden_states

        if self.sliced_heads > 0:
            input_ndim = hidden_states.ndim

            assert input_ndim == 3

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = module.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, module.heads, -1, attention_mask.shape[-1])

            if module.group_norm is not None:
                hidden_states = module.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = module.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif module.norm_cross:
                encoder_hidden_states = module.norm_encoder_hidden_states(encoder_hidden_states)

            key = module.to_k(encoder_hidden_states)
            value = module.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // module.heads

            query = query.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, module.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = F.linear(hidden_states, module.to_out[0].weight, bias=None)
            # dropout
            hidden_states = module.to_out[1](hidden_states)
        else:
            hidden_states = torch.zeros(
                [hidden_states.shape[0], hidden_states.shape[1], module.to_out[0].out_features],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        # get results using all reduce
        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, group=distri_config.batch_group, async_op=False)

        if module.to_out[0].bias is not None:
            hidden_states = hidden_states + module.to_out[0].bias.view(1, 1, -1)

        if module.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / module.rescale_output_factor

        self.counter += 1

        return hidden_states



class Conv2dTP(BaseModule):
    def __init__(self, 
                 module: nn.Conv2d, 
                 distri_config: DistriConfig):
        super(Conv2dTP, self).__init__(module, distri_config)
        assert module.in_channels % distri_config.n_device_per_batch == 0

        sharded_module = nn.Conv2d(
            module.in_channels // distri_config.n_device_per_batch,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        start_idx = distri_config.split_idx() * (module.in_channels // distri_config.n_device_per_batch)
        end_idx = (distri_config.split_idx() + 1) * (module.in_channels // distri_config.n_device_per_batch)
        sharded_module.weight.data.copy_(module.weight.data[:, start_idx:end_idx])
        if module.bias is not None:
            sharded_module.bias.data.copy_(module.bias.data)

        self.module = sharded_module
        del module

    def forward(self, x: torch.Tensor):
        distri_config = self.distri_config

        b, c, h, w = x.shape
        start_idx = distri_config.split_idx() * (c // distri_config.n_device_per_batch)
        end_idx = (distri_config.split_idx() + 1) * (c // distri_config.n_device_per_batch)
        output = F.conv2d(
            x[:, start_idx:end_idx],
            self.module.weight,
            bias=None,
            stride=self.module.stride,
            padding=self.module.padding,
            dilation=self.module.dilation,
            groups=self.module.groups,
        )
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=distri_config.batch_group, async_op=False)
        if self.module.bias is not None:
            output = output + self.module.bias.view(1, -1, 1, 1)

        self.counter += 1
        return output


class FFNTP(BaseModule):
    def __init__(self, 
                 module: FeedForward, 
                 distri_config: DistriConfig):
        super(FFNTP, self).__init__(module, distri_config)
        print(module.net[0])
        assert isinstance(module.net[0], GELU)
        assert module.net[0].proj.out_features % (distri_config.n_device_per_batch) == 0
        assert module.net[2].in_features % distri_config.n_device_per_batch == 0

        mid_features = module.net[2].in_features // distri_config.n_device_per_batch

        sharded_fc1 = nn.Linear(
            module.net[0].proj.in_features,
            mid_features,
            bias=module.net[0].proj.bias is not None,
            device=module.net[0].proj.weight.device,
            dtype=module.net[0].proj.weight.dtype,
        )

        start_idx = distri_config.split_idx() * mid_features
        end_idx = (distri_config.split_idx() + 1) * mid_features

        sharded_fc1.weight.data[:mid_features].copy_(module.net[0].proj.weight.data[start_idx:end_idx])
        if module.net[0].proj.bias is not None:
            sharded_fc1.bias.data[:mid_features].copy_(module.net[0].proj.bias.data[start_idx:end_idx])

        sharded_fc2 = nn.Linear(
            mid_features,
            module.net[2].out_features,
            bias=False,
            device=module.net[2].weight.device,
            dtype=module.net[2].weight.dtype,
        )

        self.register_parameter("bias2", nn.Parameter(module.net[2].bias.data.clone()))

        old_fc1 = module.net[0].proj
        old_fc2 = module.net[2]

        module.net[0].proj = sharded_fc1
        module.net[2] = sharded_fc2

        del old_fc1
        del old_fc2
        torch.cuda.empty_cache()

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0):

        assert scale == 1.0
        hidden_states = self.module(hidden_states)

        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, async_op=False)
        hidden_states = hidden_states + self.bias2.view(1, 1, -1)

        self.counter += 1

        return hidden_states
