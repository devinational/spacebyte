import psutil
import os
import sys
import contextlib
# import resource
import dataclasses
from typing import get_origin, get_args, Union, Optional
# import math
# from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import tiktoken
import sentencepiece as spm

# https://twitter.com/karpathy/status/1621578354024677377
# https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
DIM_MULT = 64

def interactive_mode() -> bool:
    """Check if Python is running in interactive mode.
    
    Returns:
        bool: True if running in interactive mode (e.g., Python REPL),
              False if running as a script.
    """
    return hasattr(sys, 'ps1')

def notNone(x: object) -> bool | None:
    """Check if a value is not None and return a boolean indicator.
    
    Args:
        x: Any Python object to check for None-ness
        
    Returns:
        bool | None: True if x is not None, None otherwise
    """
    return True if x is not None else None

def chrs(ts: list[int] | tuple[int, ...]) -> str:
    """Convert a sequence of integer values to a string, replacing non-printable characters.
    
    Args:
        ts: A sequence of integer values representing character codes
        
    Returns:
        str: A string where each character corresponds to an input integer,
             with non-printable characters replaced by '¤' (character 164)
    """
    def chr_(t: int) -> str:
        """Convert a single integer to a character, replacing non-printable ones.
        
        Args:
            t: Integer value representing a character code
            
        Returns:
            str: The character if it's printable (ASCII 32-126 or 161+, except 173),
                 or '¤' (character 164) for non-printable characters
        """
        return chr(t if 32 <= t <= 126 or (161 <= t and t != 173) else 164) #  or (128 <= t <= 130)
    return ''.join(chr_(t) for t in ts)

def mean2(x: torch.Tensor) -> torch.Tensor:
    """Calculate the root mean square (RMS) value of a tensor.
    
    Computes sqrt(mean(x²)) for the input tensor, which is useful for
    calculating the RMS (quadratic mean) of values in machine learning contexts.
    
    Args:
        x: Input tensor of any shape
        
    Returns:
        torch.Tensor: A scalar tensor containing the RMS value
    """
    return (x*x).mean().sqrt()

def norm2(x: torch.Tensor) -> torch.Tensor:
    """Calculate the squared L2 norm (dot product) of a tensor with itself.
    
    Computes the sum of squared elements in the tensor by flattening it and
    computing the dot product with itself: sum(x_i²) for all elements x_i.
    
    Args:
        x: Input tensor of any shape
        
    Returns:
        torch.Tensor: A scalar tensor containing the squared L2 norm
    """
    return torch.dot(x.flatten(), x.flatten())

def ceil_div(n: int, k: int) -> int:
    return (n+k-1)//k

def ceil(n: int, k: int) -> int:
    return ceil_div(n, k)*k

def int_div(x, y):
    div, rem = divmod(x, y)
    assert rem == 0
    return div

def is_pow2(n:int):
    return n.bit_count() == 1

class MeanError:
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.sum_squares = 0

    def add(self, x):
        self.n += 1
        self.sum += x
        self.sum_squares += x*x

    def mean(self):
        return self.sum / self.n

    def error(self):
        n = self.n
        if n == 0:
            return float('nan') * self.sum
        err = (self.sum_squares/n - (self.sum/n)**2) / (n-1)
        if isinstance(err, float):
            err = max(0, err)
        elif isinstance(err, torch.Tensor):
            err = err.clamp(min=0)
        elif isinstance(err, np.ndarray):
            err = np.clip(err, 0, None)
        else:
            print(type(err))
            assert False
        return err**0.5

byte_BOS = 255

class Tokenizer:
    tiktoken_encodings = ['gpt2', 'cl100k_base']

    def __init__(self, tokenizer: str = None):
        self.name = tokenizer

        if tokenizer is None:
            self.tokenizer = None
            self.vocab_size = 256
            self.BOS = byte_BOS
            self.file_suffix = '.txt'
        elif tokenizer in Tokenizer.tiktoken_encodings:
            self.tokenizer = tiktoken.get_encoding(tokenizer)
            self.vocab_size = self.tokenizer.n_vocab
            self.BOS = self.tokenizer.eot_token
            self.file_suffix = '.' + tokenizer
        else:
            self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer + '.model')
            self.vocab_size = self.tokenizer.get_piece_size()
            self.BOS = self.tokenizer.piece_to_id('<s>')
            self.file_suffix = '.' + tokenizer.split('/')[-1]

        assert self.vocab_size <= 2**32
        self.dtype = np.uint8  if self.vocab_size <= 2**8  else \
                     np.uint16 if self.vocab_size <= 2**16 else np.uint32

    def encode(self, text, prepend_BOS=False, dtype=torch.int64, tensor=torch.tensor, **kwargs):
        if self.tokenizer is not None:
            ret = self.tokenizer.encode_ordinary(text) if hasattr(self.tokenizer, 'encode_ordinary') else \
                self.tokenizer.encode(text) # for sentencepiece
            if prepend_BOS:
                ret = [self.BOS] + ret
        else:
            ret = text.encode('utf-8')
            if prepend_BOS:
                ret = bytes([self.BOS]) + ret
            ret = bytearray(ret)
        return tensor(ret, dtype=dtype, **kwargs)

    def decode(self, tokens):
        if self.tokenizer is not None:
            return self.tokenizer.decode(list(tokens))
        else:
            return tensor_to_str(tokens, errors='replace')

# kwargs example: errors='replace'
def tensor_to_str(x, **kwargs):
    return x.cpu().to(torch.uint8).numpy().tobytes().decode('utf-8', **kwargs)

def autocast_context(dtype):
    # torch.backends.cuda.enable_mem_efficient_sdp(False)
    # torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    if isinstance(dtype, str):
        dtype = eval(f'torch.{dtype}')
    return torch.amp.autocast(device_type='cuda', dtype=dtype) \
        if dtype != torch.float32 else contextlib.nullcontext()

def get_memory_stats(*, device):
    ret = ''.join([
        f'{psutil.Process(os.getpid()).memory_info().rss/2**30:.1f}GB RAM',
        f', {torch.cuda.max_memory_allocated()/2**30:.1f}GB cuda' if 'cuda' in device else '',
        f', {torch.mps.driver_allocated_memory()/2**30:.1f}GB mps' if 'mps' in device else '' ])
    if 'cuda' in device:
        torch.cuda.reset_peak_memory_stats()
    return ret

def like(tensor: torch.Tensor):
    return {'dtype': tensor.dtype, 'device': tensor.device}

def num_params(module):
    if isinstance(module, nn.Module):
        return sum(p.numel() for p in module.parameters())
    else:
        return module.numel()

def log_forward(forward):
    def logged_forward(self, x, *args, log=None, **kwargs):
        y = forward(self, x, *args, log=log, **kwargs)
        # if log is not None:
            # log[f'{self.module_name}.x'] = mean2(x.detach())
            # log[f'{self.module_name}.y'] = mean2(y.detach())
        return y
    return logged_forward

def tensor_items(xs, dtype=None):
    if isinstance(xs, list):
        return [tensor_items(x) for x in xs]
    elif isinstance(xs, tuple):
        return tuple(tensor_items(x) for x in xs)
    elif isinstance(xs, dict):
        return {k: tensor_items(v) for k,v in xs.items()}
    elif isinstance(xs, torch.Tensor):
        if dtype is None:
            dtype = xs.dtype
            if dtype == torch.bfloat16:
                dtype = torch.float32
        return xs.item() if xs.dim()==0 else xs.detach().cpu().to(dtype=dtype).numpy()
    elif isinstance(xs, np.ndarray):
        return xs if dtype is None else xs.astype(dtype, copy=True)
    elif hasattr(xs, '__next__') and not hasattr(xs, '__getitem__'):
        return (tensor_items(x) for x in xs)
    else:
        return xs

def default_device():
    return 'cuda' if torch.cuda.is_available() else \
           'mps'  if torch.backends.mps.is_available() else 'cpu'

def synchronize_device(device):
    device = torch.device(device).type
    eval(f'torch.{device}.synchronize')()

def empty_cache(device):
    def device_is(dev):
        return dev in device if isinstance(device, str) else dev == torch.device(device).type

    if device_is('cuda'):
        torch.cuda.empty_cache()
    elif device_is('mps'):
        torch.mps.empty_cache()

from typing import Optional, get_origin, get_args

def make_dataclasses(data_classes, **kwargs):
    field_typess = [ {field.name : field.type for field in dataclasses.fields(data_class)}
        for data_class in data_classes ]
    dicts = [{} for _ in data_classes]

    def convert_value(value, field_type):
        if value is None:
            return None
            
        origin = get_origin(field_type)
        if origin is Union:
            types = get_args(field_type)
            # Handle Optional[T] which is Union[T, None]
            if type(None) in types and len(types) == 2:
                actual_type = next(t for t in types if t is not type(None))
                return convert_value(value, actual_type)
            # Try each possible type until one works
            for t in types:
                try:
                    return convert_value(value, t)
                except (ValueError, TypeError):
                    continue
            raise ValueError(f"Could not convert {value} to any of {types}")
        elif origin is not None:
            # Handle other generic types (List, Dict, etc)
            args = get_args(field_type)
            if origin is list:
                return [convert_value(v, args[0]) for v in value]
            elif origin is dict:
                return {k: convert_value(v, args[1]) for k, v in value.items()}
            # Add other container types as needed
            return value
        else:
            # Basic type conversion
            return field_type(value)

    for k, v in kwargs.items():
        used = False
        for field_types, dict0 in zip(field_typess, dicts):
            field_type = field_types.get(k)
            if field_type is not None:
                assert not used
                used = True
                try:
                    dict0[k] = convert_value(v, field_type)
                except Exception as e:
                    raise ValueError(f"Failed to convert {k}={v} to {field_type}: {str(e)}")
        
        if not used:
            raise Exception(f'make_dataclasses: {k} not found in {data_classes}')

    return [data_class(**dict0) for data_class, dict0 in zip(data_classes, dicts)]

# def entropy(logits):
    # return - (logits.softmax(-1) * logits.log_softmax(-1)).sum(-1) # todo inner product

def cross_entropy(logits, targets, reduction='mean', ignore_index=None):
        B, T, V = logits.shape

        if reduction == 'mean' and ignore_index is None:
            return F.cross_entropy(logits.reshape(B*T, V), targets.reshape(B*T), reduction='mean')

        cross_entropy = F.cross_entropy(logits.reshape(B*T, V), targets.reshape(B*T),
            ignore_index=ignore_index if ignore_index is not None else -100,
            reduction='none').view(B, T)
        if reduction == 'none':
            return cross_entropy

        if reduction == 'batch':
            return cross_entropy.sum(0) / (targets >= 0).sum(0) # T
        
        # NOTE: pytorch takes the mean over batch and context at the same time,
        # which isn't invariant under changes of micro-batch size when some indices are ignored
        # to fix this, we take the mean over the context before taking a separate mean over the batch

        cross_entropy = cross_entropy.sum(1) / (targets >= 0).sum(1) # B
        if reduction == 'context':
            return cross_entropy
        
        assert reduction == 'mean'
        return cross_entropy.mean()

# Tee

class Tee:
    def __init__(self, file_name):
        self.file = None
        if Tee._TeeStreams is None:
            # do nothing if sys.stdout or sys.stderr have been modified (e.g. in jupyter)
            if sys.stdout is not sys.__stdout__ or sys.stderr is not sys.__stderr__:
                return
            sys.stdout = Tee._TeeStream(sys.stdout)
            sys.stderr = Tee._TeeStream(sys.stderr)
            Tee._TeeStreams = (sys.stdout, sys.stderr)
        
        self.file = open(file_name, 'w', buffering=1)
        Tee._files.append(self.file)

    _files = []
    _TeeStreams = None

    def __del__(self):
        if self.file is not None:
            self.file.close()
            Tee._files.remove(self.file)
            self.file = None

    class _TeeStream:
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            self.stream.write(data)
            for f in Tee._files:
                f.write(data)

        def flush(self):
            self.stream.flush()
            for f in Tee._files:
                f.flush()
