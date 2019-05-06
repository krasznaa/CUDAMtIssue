# CUDA + TBB Issue

## Building The Code

In order to build the code you'll need:
  - [CMake](https://cmake.org/) version >= 3.13;
  - A recent version of [Boost](https://www.boost.org/);
  - A recent version of [TBB](https://www.threadingbuildingblocks.org/);
  - And of course a recent version of CUDA. I've been using version 10.1.

## The Symptom

This is a simplified version of a piece of code that I started to develop
for my work. What I found with it is that once I send "large enough" tasks
to my GPU from multiple TBB tasks with it, the GPU kernel would start
misbehaving. I reported about this issue in:

https://devtalk.nvidia.com/default/topic/1051078/cuda-programming-and-performance/how-to-avoid-overcommitting-a-gpu-/

The issue is demonstrated by the
[test_CalibrateParticles.cu](tests/testCalibrateParticles.cu) executable of
the repository. When running the executable with a "small enough" payload, it
works just fine.

```
[bash][atspot01]:build_mt > ./tests/test_CalibrateParticles -v 100 -p 1000 -i 200 -t 8 -g 4 -s 5
Using:
  - Max. Particles : 1000
  - Calc-iterations: 200
  - Events         : 100
  - CPU Threads    : 8
  - GPU Tasks      : 4
  - GPU Streams    : 5
Processed 0 / 100 events
[bash][atspot01]:build_mt >
```

But once I increase the amount of calculations to a certain level, the code
ends up failing like:

```
[bash][atspot01]:build_mt > ./tests/test_CalibrateParticles -v 100 -p 10000 -i 2000 -t 8 -g 4 -s 5
Using:
  - Max. Particles : 10000
  - Calc-iterations: 2000
  - Events         : 100
  - CPU Threads    : 8
  - GPU Tasks      : 4
  - GPU Streams    : 5
terminate called after throwing an instance of 'std::runtime_error'
  what():  /home/krasznaa/projects/cuda/CUDAMtIssue/core/Memory.cu:70 Failed to execute: cudaFreeHost( info.ptr ) (an illegal memory access was encountered)
Aborted (core dumped)
[bash][atspot01]:build_mt > 
```

Looking at the failure with
[cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/index.html) I find that
an invalid memory access happens in all of these cases. Although the exact
failure is pretty random. I've even seen the debugger point at the destructor
of an object that was created **on the stack**, for an invalid memory access.

```
[bash][atspot01]:build_mt > cuda-gdb ./tests/test_CalibrateParticles
NVIDIA (R) CUDA Debugger
10.1 release
Portions Copyright (C) 2007-2018 NVIDIA Corporation
GNU gdb (GDB) 7.12
Copyright (C) 2016 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
and "show warranty" for details.
This GDB was configured as "x86_64-pc-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
<http://www.gnu.org/software/gdb/documentation/>.
For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./tests/test_CalibrateParticles...done.
(cuda-gdb) run -v 100 -p 10000 -i 2000 -t 8 -g 4 -s 5
Starting program: /afs/cern.ch/work/k/krasznaa/CUDA/build_mt/tests/test_CalibrateParticles -v 100 -p 10000 -i 2000 -t 8 -g 4 -s 5
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib64/libthread_db.so.1".
Using:
  - Max. Particles : 10000
  - Calc-iterations: 2000
  - Events         : 100
  - CPU Threads    : 8
  - GPU Tasks      : 4
  - GPU Streams    : 5
[New Thread 0x7fffeef92700 (LWP 9844)]
[New Thread 0x7fffee353700 (LWP 9845)]
[New Thread 0x7fffedb52700 (LWP 9846)]
[New Thread 0x7fffed2d0700 (LWP 9847)]
[New Thread 0x7fffec78c700 (LWP 9848)]
[New Thread 0x7fffe32c7700 (LWP 9850)]
[New Thread 0x7fffe36c8700 (LWP 9849)]
[New Thread 0x7fffe2ec6700 (LWP 9851)]
[New Thread 0x7fffe26c4700 (LWP 9853)]
[New Thread 0x7fffe2ac5700 (LWP 9852)]
[New Thread 0x7fffe22c3700 (LWP 9854)]
[New Thread 0x7fffe1ec2700 (LWP 9855)]
[New Thread 0x7fffe16c1700 (LWP 9857)]

CUDA Exception: Warp Illegal Address

Thread 3 "test_CalibrateP" received signal CUDA_EXCEPTION_14, Warp Illegal Address.
[Switching focus to CUDA kernel 1, grid 4, block (0,0,0), thread (32,0,0), device 0, sm 13, warp 15, lane 0]
0x00007fffa42afe50 in (anonymous namespace)::ParticleCalibrator::operator() (this=0x7ffff2fffb10, index=32, 
    aux=0x7ffff2fffb18, etaId=1, phiId=2, ptId=0, iterations=2000)
    at /home/krasznaa/projects/cuda/CUDAMtIssue/tests/kernels/CalibrateParticles.cu:23
23      /home/krasznaa/projects/cuda/CUDAMtIssue/tests/kernels/CalibrateParticles.cu: No such file or directory.
(cuda-gdb) bt
#0  0x00007fffa42afe50 in (anonymous namespace)::ParticleCalibrator::operator() (this=0x7ffff2fffb10, index=32, 
    aux=0x7ffff2fffb18, etaId=1, phiId=2, ptId=0, iterations=2000)
    at /home/krasznaa/projects/cuda/CUDAMtIssue/tests/kernels/CalibrateParticles.cu:23
#1  0x00007fffa42c9240 in (anonymous namespace)::deviceKernel<(anonymous namespace)::ParticleCalibrator, unsigned long, unsigned long, unsigned long, unsigned long><<<(22,1,1),(256,1,1)>>> (csize=5488, vsize=3, vars=0x7fffc7437800, 
    args=1, args=1, args=1, args=1) at /home/krasznaa/projects/cuda/CUDAMtIssue/kernel/KernelRun.icc:66
(cuda-gdb)
```

When I run `cuda-gdb` on a "succeeding configuration" (small enough GPU
payload), it doesn't report anything suspicious. Furthermore,
[cuda-memcheck](https://docs.nvidia.com/cuda/cuda-memcheck/index.html) has no
clue of what could be going wrong. It can't identify any problematic memory
access in my code.

```
[bash][atspot01]:build_mt > cuda-memcheck ./tests/test_CalibrateParticles -v 100 -p 10000 -i 2000 -t 8 -g 4 -s 5
========= CUDA-MEMCHECK
Using:
  - Max. Particles : 10000
  - Calc-iterations: 2000
  - Events         : 100
  - CPU Threads    : 8
  - GPU Tasks      : 4
  - GPU Streams    : 5
Processed 0 / 100 events
========= ERROR SUMMARY: 0 errors
[bash][atspot01]:build_mt >
```

Running the test through `cuda-memcheck` results in it running **much** slower,
but it does finish successfully.

And finally, what made me create this standalone example: When I use this
code on a laptop with an
[MX150 GPU](https://www.geforce.com/hardware/notebook-gpus/geforce-mx150),
I can not make the code fail. :confused: This test would always finish
successfully on that laptop.

For this reason by now I suspect something going wrong in CUDA in this
test...
