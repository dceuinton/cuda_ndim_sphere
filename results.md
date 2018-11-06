# Results 

===========================================================

CSV Copy and Paste
TPB, BPG, dimensions, radius, total points in sphere, time for kernel to run (ms),
256,1,1,25.5000,51,0.0538,
256,1,2,2.0500,13,0.0496,
256,1,3,1.5000,19,0.0525,
256,1,1,2.0500,5,0.0501,
256,1,1,5.0500,11,0.0507,
256,1,1,10.0500,21,0.0470,
256,1,2,2.0500,13,0.0499,
256,1,2,5.0500,81,0.0516,
256,2,2,10.0500,325,0.0588,
256,1,3,2.0500,33,0.0552,
256,6,3,5.0500,515,0.0737,
256,37,3,10.0500,4337,0.4360,
256,3,4,2.0500,89,0.0741,
256,58,4,5.0500,3121,0.3118,
256,760,4,10.0500,50505,3.3434,
256,13,5,2.0500,221,0.1163,
256,630,5,5.0500,16875,2.8302,
256,15954,5,10.0500,543149,64.1055,
256,62,6,2.0500,485,0.3304,
256,6921,6,5.0500,84769,27.9263,
256,335024,6,10.0500,0,130.2936,
256,306,7,2.0500,953,1.4403,
256,76122,7,5.0500,0,30.0821,
256,1526,8,2.0500,1713,6.6078,
256,837340,8,5.0500,0,326.9352

===========================================================

# Device Query

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GTX 470"
  CUDA Driver Version / Runtime Version          7.5 / 7.5
  CUDA Capability Major/Minor version number:    2.0
  Total amount of global memory:                 1279 MBytes (1341325312 bytes)
  (14) Multiprocessors, ( 32) CUDA Cores/MP:     448 CUDA Cores
  GPU Max Clock rate:                            1215 MHz (1.22 GHz)
  Memory Clock rate:                             1674 Mhz
  Memory Bus Width:                              320-bit
  L2 Cache Size:                                 655360 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65535), 3D=(2048, 2048, 2048)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 32768
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (65535, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 7.5, CUDA Runtime Version = 7.5, NumDevs = 1, Device0 = GeForce GTX 470
Result = PASS