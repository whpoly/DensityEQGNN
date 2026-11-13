"""Frozen-weight radial interpolation and an inference-only fused CSR kernel.

NVRTC/driver loading follows NVIDIA's documented compile-PTX/load/launch flow:
https://docs.nvidia.com/cuda/nvrtc/index.html
No installed compiler or third-party GPU Python package is needed on Windows.
"""
import ctypes as ct
import os
from pathlib import Path

import torch

from scripts.deepdft_factored_messages import FactoredProbe

CUDA_SOURCE = Path(__file__).with_name('radial_message_kernel.cu')


class RadialKernel:
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('This local NVRTC loader currently supports Windows only')
        torch.cuda.init()
        library = Path(torch.__file__).parent/'lib'
        self.dll_directory = os.add_dll_directory(str(library))
        self.nvrtc = ct.WinDLL(str(library/'nvrtc64_120_0.dll'))
        self.driver = ct.WinDLL('nvcuda.dll')
        pointer = ct.c_void_p
        for name,args in {
            'nvrtcCreateProgram':[ct.POINTER(pointer),ct.c_char_p,ct.c_char_p,ct.c_int,pointer,pointer],
            'nvrtcCompileProgram':[pointer,ct.c_int,ct.POINTER(ct.c_char_p)],
            'nvrtcGetProgramLogSize':[pointer,ct.POINTER(ct.c_size_t)],
            'nvrtcGetProgramLog':[pointer,pointer],
            'nvrtcGetPTXSize':[pointer,ct.POINTER(ct.c_size_t)],
            'nvrtcGetPTX':[pointer,pointer],
            'nvrtcDestroyProgram':[ct.POINTER(pointer)],
        }.items():
            getattr(self.nvrtc,name).argtypes = args
            getattr(self.nvrtc,name).restype = ct.c_int
        for name,args in {
            'cuModuleLoadData':[ct.POINTER(pointer),pointer],
            'cuModuleGetFunction':[ct.POINTER(pointer),pointer,ct.c_char_p],
            'cuLaunchKernel':[pointer,*([ct.c_uint]*7),pointer,ct.POINTER(pointer),pointer],
            'cuGetErrorString':[ct.c_int,ct.POINTER(ct.c_char_p)],
        }.items():
            getattr(self.driver,name).argtypes = args
            getattr(self.driver,name).restype = ct.c_int
        program = pointer()
        self.check(self.nvrtc.nvrtcCreateProgram(ct.byref(program),CUDA_SOURCE.read_bytes(),
            b'radial_message_kernel.cu',0,None,None),'create program')
        major,minor = torch.cuda.get_device_capability()
        self.options = [f'--gpu-architecture=compute_{major}{minor}','--std=c++14','--fmad=false']
        options = (ct.c_char_p*len(self.options))(*[s.encode() for s in self.options])
        status = self.nvrtc.nvrtcCompileProgram(program,len(options),options)
        size = ct.c_size_t()
        self.nvrtc.nvrtcGetProgramLogSize(program,ct.byref(size))
        log = ct.create_string_buffer(size.value)
        self.nvrtc.nvrtcGetProgramLog(program,log)
        self.compile_log = log.value.decode()
        if status:
            raise RuntimeError(f'NVRTC error {status}: {self.compile_log}')
        self.check(self.nvrtc.nvrtcGetPTXSize(program,ct.byref(size)),'PTX size')
        ptx = ct.create_string_buffer(size.value)
        self.check(self.nvrtc.nvrtcGetPTX(program,ptx),'PTX')
        self.nvrtc.nvrtcDestroyProgram(ct.byref(program))
        self.module,self.function = pointer(),pointer()
        self.check(self.driver.cuModuleLoadData(ct.byref(self.module),ptx),'load module',True)
        self.check(self.driver.cuModuleGetFunction(ct.byref(self.function),self.module,b'radial_messages'),'get function',True)

    def check(self,status,label,driver=False):
        if status:
            message = ct.c_char_p()
            if driver:
                self.driver.cuGetErrorString(status,ct.byref(message))
            raise RuntimeError(f'{label}: {status} {message.value}')

    def __call__(self,offsets,source,distance,direction,envelope,table,s,v,p,size,cutoff):
        if torch.is_grad_enabled():
            raise RuntimeError('Fused kernel is inference-only; use torch.no_grad()')
        channels = s.shape[-1]
        if not 0 < channels <= 1024:
            raise ValueError('Unsupported channel count')
        values = [offsets,source,distance,direction,envelope,table,s,v,p]
        if any(not x.is_cuda or not x.is_contiguous() or x.device!=s.device for x in values):
            raise ValueError('Contiguous CUDA tensors on a single device required')
        if any(x.dtype!=torch.int64 for x in values[:2]) or any(x.dtype!=torch.float32 for x in values[2:]):
            raise ValueError('Kernel requires int64 indices and float32 values')
        out_s,out_v = s.new_empty(size,channels),s.new_empty(size,3,channels)
        if not size:
            return out_s,out_v
        arguments = [ct.c_void_p(x.data_ptr()) for x in (*values,out_s,out_v)]
        arguments += [ct.c_int(size),ct.c_int(channels),ct.c_int(len(table)-1),ct.c_float(cutoff)]
        pointers = (ct.c_void_p*len(arguments))(*[ct.cast(ct.byref(x),ct.c_void_p) for x in arguments])
        with torch.cuda.device(s.device):
            stream = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
            status = self.driver.cuLaunchKernel(self.function,size,1,1,channels,1,1,0,stream,pointers,None)
        self.check(status,'launch kernel',True)
        return out_s,out_v


class RadialTableProbe(FactoredProbe):
    def __init__(self,model,points=16385,fused=False):
        super().__init__(model)
        if points < 2:
            raise ValueError('At least two table points required')
        self.points = points
        self._layer_index = {id(layer):k for k,layer in enumerate(self.probe.message_layers)}
        with torch.no_grad():
            device = next(model.parameters()).device
            r = torch.linspace(0,self.probe.cutoff,points,device=device)
            diff = torch.stack((r,r*0,r*0),1)
            _,radial,envelope = super().geometry(diff)
            tables = torch.stack([layer.filter_layer(radial)*envelope for layer in self.probe.message_layers])
            self.register_buffer('tables',tables)
        self.kernel = RadialKernel() if fused else None

    def geometry(self,diff):
        distance = diff.square().sum(1,keepdim=True).sqrt()
        direction = diff/torch.linalg.vector_norm(diff,dim=1,keepdim=True).clamp_min(1e-12)
        return direction.contiguous(),distance,torch.ones_like(distance)

    def lookup(self,k,distance):
        x = distance.flatten().clamp(0,self.probe.cutoff)*((self.points-1)/self.probe.cutoff)
        lo = x.long().clamp(max=self.points-2)
        t = (x-lo.float())[:,None]
        table = self.tables[k]
        out = table[lo]*(1-t)+table[lo+1]*t
        return out*((distance>=0)&(distance<self.probe.cutoff))

    def aggregate(self,layer,scalar,vector,projection,source,receiver,geometry,size):
        direction,distance,envelope = geometry
        k = self._layer_index[id(layer)]
        if self.kernel is not None:
            # Cache only within this freshly generated geometry tensor. Never
            # reuses learned states or geometry across calls/structures.
            if not hasattr(distance,'_csr_offsets'):
                counts = torch.bincount(receiver,minlength=size)
                distance._csr_offsets = torch.cat((counts.new_zeros(1),counts.cumsum(0)))
            return self.kernel(distance._csr_offsets,source,distance,direction,envelope,self.tables[k],
                scalar,vector,projection,size,self.probe.cutoff)
        fv,fe,fs = (self.lookup(k,distance)*envelope).split(scalar.shape[-1],1)
        ms = scalar[source]*fs
        mv = vector[source]*fv[:,None]+(projection[source]*fe)[:,None]*direction[:,:,None]
        a = scalar.new_zeros(size,scalar.shape[-1]).index_add_(0,receiver,ms)
        b = vector.new_zeros(size,3,scalar.shape[-1]).index_add_(0,receiver,mv)
        return a,b
