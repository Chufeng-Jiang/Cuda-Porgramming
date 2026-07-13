- [外部循环，内部调用kernel](#外部循环内部调用kernel)
  - [编译运行](#编译运行)
  - [查看 kernel 统计：](#查看-kernel-统计)
  - [查看 CUDA API 调用：](#查看-cuda-api-调用)
  - [查看完整汇总：](#查看完整汇总)
- [grid-stride loop 先调用kernel, 然后进行内部循环](#grid-stride-loop-先调用kernel-然后进行内部循环)
  - [编译运行](#编译运行-1)
  - [查看 kernel 统计：](#查看-kernel-统计-1)
  - [查看 CUDA API 调用：](#查看-cuda-api-调用-1)
  - [查看完整汇总：](#查看完整汇总-1)
- [CUDA-GDB](#cuda-gdb)
  - [CUDA-GDB 支持 WSL2，但需要在 Windows 中把 EnableInterface 设置为 DWORD 1](#cuda-gdb-支持-wsl2但需要在-windows-中把-enableinterface-设置为-dword-1)
  - [编译运行](#编译运行-2)
  - [查询函数](#查询函数)
  - [打断点](#打断点)
  - [运行](#运行)


# 外部循环，内部调用kernel

```
// CUDA 加法 kernel
template <typename T>
__global__ void add_kernel(
    T* c,
    const T* a,
    const T* b,
    std::size_t n,
    std::size_t offset
) {
    std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x
        + offset;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}


// 使用固定 grid/block 配置完成整个向量加法
template <typename T>
void vector_add(
    T* c,
    const T* a,
    const T* b,
    std::size_t n,
    const dim3& grid,
    const dim3& block
) {
    // 每次 kernel launch 最多处理的元素数量
    std::size_t step =
        static_cast<std::size_t>(grid.x) * block.x;

    // 外层循环保证全部 n 个元素都被处理
    for (std::size_t offset = 0; offset < n; offset += step) {
        // 内部启动 kernel 处理当前 step 个元素
        add_kernel<T><<<grid, block>>>(
            c,
            a,
            b,
            n,
            offset
        );

        CUDA_CHECK(cudaGetLastError());
    }
}


```

## 编译运行

```
nvcc -O3 -std=c++17 vector_add.cu -o vector_add

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=vector_add \
  --force-overwrite=true \
  ./vector_add
```



## 查看 kernel 统计：

```
(base) chufeng@Chufeng:~/Desktop/InfiniTensor/coursedemo/cuda/01$ nsys stats \
  --force-export=true \
  --report cuda_gpu_kern_sum \
  vector_add.nsys-rep


Generating SQLite file vector_add.sqlite from vector_add.nsys-rep
Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                         Name                                       
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------
    100.0      21570333587   16777216    1285.7    1248.0      1184      5120         67.6  void add_kernel<float>(T1 *, const T1 *, const T1 *, unsigned long, unsigned long)
```

## 查看 CUDA API 调用：

```
(base) chufeng@Chufeng:~/Desktop/InfiniTensor/coursedemo/cuda/01$ nsys stats \
  --force-export=true \
  --report cuda_api_sum \
  vector_add.nsys-rep

NOTICE: Existing SQLite export found: vector_add.sqlite
It is assumed file was previously exported from: vector_add.nsys-rep
Consider using --force-export=true if needed.

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/cuda_api_sum.py]...

** CUDA API Summary (cuda_api_sum):

Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)          Name

---

99.8     115231828203   16777216      6868.4     4058.0      2619    9735564      59230.1  cudaLaunchKernel
0.1        162742230          3  54247410.0   355937.0    316923  162069370   93376558.5  cudaMalloc
0.0         19682616          2   9841308.0  9841308.0   7171463   12511153    3775731.0  cudaMemcpy
0.0          1773413          2    886706.5   886706.5       905    1772508    1252712.5  cudaEventCreate
0.0           936608          3    312202.7   215351.0    178368     542889     200634.2  cudaFree
0.0           299856          2    149928.0   149928.0     38872     260984     157056.9  cudaEventRecord
0.0           208631          1    208631.0   208631.0    208631     208631          0.0  cudaEventSynchronize
0.0            19133          2      9566.5     9566.5      1041      18092      12056.9  cudaEventDestroy
```

## 查看完整汇总：

```
(base) chufeng@Chufeng:~/Desktop/InfiniTensor/coursedemo/cuda/01$ nsys stats vector_add.nsys-rep

NOTICE: Existing SQLite export found: vector_add.sqlite
        It is assumed file was previously exported from: vector_add.nsys-rep
        Consider using --force-export=true if needed.

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/nvtx_sum.py]... 
SKIPPED: vector_add.sqlite does not contain NV Tools Extension (NVTX) data.

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/osrt_sum.py]... 

 ** OS Runtime Summary (osrt_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)  Max (ns)   StdDev (ns)          Name         
 --------  ---------------  ---------  -----------  -----------  --------  ---------  -----------  ---------------------
     81.6     244332940321       2435  100342069.9  100163858.0      2208  137091060    4818634.4  poll                 
     18.0      53873840694   15771376       3415.9       1249.0       999   27350272      60156.2  ioctl                
      0.4       1145063576       1468     780016.1     128836.0     44607    9568093    1375923.4  pthread_rwlock_wrlock
      0.0         99703479       1303      76518.4      65688.0      1006    3005241     151000.4  pthread_rwlock_rdlock
      0.0         40847296          6    6807882.7     258209.5      1700   39436525   15986442.4  fread                
      0.0          3916616         10     391661.6       2390.5      1114    3859267    1218422.7  fclose               
      0.0          3165201         26     121738.5       3374.0      1164    1254827     333288.4  fopen                
      0.0           630436          3     210145.3      70476.0     52724     507236     257441.1  sem_timedwait        
      0.0           610114          3     203371.3       2782.0      2372     604960     347786.0  fwrite               
      0.0           489049          4     122262.3     123692.0     82034     159631      32656.8  pthread_create       
      0.0           340913          5      68182.6       8300.0      4138     315137     138065.4  open                 
      0.0           249322         22      11332.8       7130.0      2022      32551       9057.1  mmap                 
      0.0           136290          1     136290.0     136290.0    136290     136290          0.0  pthread_join         
      0.0            74306         10       7430.6       3914.0      1327      25144       7768.8  fgets                
      0.0            56779          4      14194.8      15264.5      8994      17256       3593.4  write                
      0.0            26305          1      26305.0      26305.0     26305      26305          0.0  putc                 
      0.0            15742          6       2623.7       2451.0      1338       5041       1317.1  read                 
      0.0            11520          3       3840.0       3928.0      2371       5221       1427.0  pipe2                
      0.0            10026          5       2005.2       1846.0      1249       3294        815.0  fcntl                
      0.0             2460          2       1230.0       1230.0      1124       1336        149.9  fflush               
      0.0             1484          1       1484.0       1484.0      1484       1484          0.0  close                

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)          Name        
 --------  ---------------  ---------  ----------  ---------  --------  ---------  -----------  --------------------
     99.8     115231828203   16777216      6868.4     4058.0      2619    9735564      59230.1  cudaLaunchKernel    
      0.1        162742230          3  54247410.0   355937.0    316923  162069370   93376558.5  cudaMalloc          
      0.0         19682616          2   9841308.0  9841308.0   7171463   12511153    3775731.0  cudaMemcpy          
      0.0          1773413          2    886706.5   886706.5       905    1772508    1252712.5  cudaEventCreate     
      0.0           936608          3    312202.7   215351.0    178368     542889     200634.2  cudaFree            
      0.0           299856          2    149928.0   149928.0     38872     260984     157056.9  cudaEventRecord     
      0.0           208631          1    208631.0   208631.0    208631     208631          0.0  cudaEventSynchronize
      0.0            19133          2      9566.5     9566.5      1041      18092      12056.9  cudaEventDestroy    

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                         Name                                       
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------
    100.0      21570333587   16777216    1285.7    1248.0      1184      5120         67.6  void add_kernel<float>(T1 *, const T1 *, const T1 *, unsigned long, unsigned long)

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/cuda_gpu_mem_time_sum.py]... 

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation          
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
    100.0         13057136      2  6528568.0  6528568.0   5981185   7075951     774116.5  [CUDA memcpy Host-to-Device]

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/cuda_gpu_mem_size_sum.py]... 

 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
    134.218      2    67.109    67.109    67.109    67.109        0.000  [CUDA memcpy Host-to-Device]


```


# grid-stride loop 先调用kernel, 然后进行内部循环

```
template <typename T>
__global__ void add_kernel_inner_loop(
    T* c,
    const T* a,
    const T* b,
    std::size_t n,
    std::size_t step
) {
    std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    for (std::size_t i = idx; i < n; i += step) {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
void vector_add_inner_loop(
    T* c,
    const T* a,
    const T* b,
    std::size_t n,
    const dim3& grid,
    const dim3& block
) {
    std::size_t step =
        static_cast<std::size_t>(grid.x) * block.x;

    add_kernel_inner_loop<T><<<grid, block>>>(
        c,
        a,
        b,
        n,
        step
    );

    CUDA_CHECK(cudaGetLastError());
}


```

## 编译运行

```
nvcc -O3 -std=c++17 vector_add.cu -o vector_add

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=vector_add \
  --force-overwrite=true \
  ./vector_add
```



## 查看 kernel 统计：

```
(base) chufeng@Chufeng:~/Desktop/InfiniTensor/coursedemo/cuda/01$ nsys stats \
  --force-export=true \
  --report cuda_gpu_kern_sum \
  vector_add.nsys-rep
Generating SQLite file vector_add.sqlite from vector_add.nsys-rep
Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)                                              Name                                             
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ---------------------------------------------------------------------------------------------
    100.0       1362203062          1  1362203062.0  1362203062.0  1362203062  1362203062          0.0  void add_kernel_inner_loop<float>(T1 *, const T1 *, const T1 *, unsigned long, unsigned long)
```

## 查看 CUDA API 调用：

```
(base) chufeng@Chufeng:~/Desktop/InfiniTensor/coursedemo/cuda/01$ nsys stats \
  --force-export=true \
  --report cuda_api_sum \
  vector_add.nsys-rep
Generating SQLite file vector_add.sqlite from vector_add.nsys-rep
Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)          Name        
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  --------------------
     89.3       1362349202          1  1362349202.0  1362349202.0  1362349202  1362349202          0.0  cudaEventSynchronize
      9.3        142073416          3    47357805.3      455766.0      243785   141373865   81420365.0  cudaMalloc          
      1.2         18745642          2     9372821.0     9372821.0     6176381    12569261    4520448.8  cudaMemcpy          
      0.1           802634          3      267544.7      207634.0      192270      402730     117325.7  cudaFree            
      0.0           495743          1      495743.0      495743.0      495743      495743          0.0  cudaLaunchKernel    
      0.0           456570          2      228285.0      228285.0        1182      455388     321172.1  cudaEventCreate     
      0.0           365509          2      182754.5      182754.5       10651      354858     243391.1  cudaEventRecord     
      0.0             5626          2        2813.0        2813.0         397        5229       3416.7  cudaEventDestroy    

```

## 查看完整汇总：

```
(base) chufeng@Chufeng:~/Desktop/InfiniTensor/coursedemo/cuda/01$ nsys stats vector_add.nsys-rep

NOTICE: Existing SQLite export found: vector_add.sqlite
        It is assumed file was previously exported from: vector_add.nsys-rep
        Consider using --force-export=true if needed.

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/nvtx_sum.py]... 
SKIPPED: vector_add.sqlite does not contain NV Tools Extension (NVTX) data.

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/osrt_sum.py]... 

 ** OS Runtime Summary (osrt_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)    Min (ns)  Max (ns)   StdDev (ns)       Name     
 --------  ---------------  ---------  ----------  -----------  --------  ---------  -----------  --------------
     96.1       2907095893         34  85502820.4  100164378.5      4129  100425281   31944931.9  poll          
      3.3        101152621        588    172028.3       4381.0      1012   19591628     991338.6  ioctl         
      0.5         14428550          5   2885710.0     125087.0     10407   13823421    6116724.1  fread         
      0.0          1215689         25     48627.6       4345.0      1193     424218     119389.9  fopen         
      0.0           838750          3    279583.3      64783.0     44022     729945     390162.8  sem_timedwait 
      0.0           348805          4     87201.3      80190.5     65506     122918      24906.2  pthread_create
      0.0           315386          5     63077.2       6486.0      3804     293471     128799.9  open          
      0.0           245460         10     24546.0       1688.5      1035     188389      58293.1  fclose        
      0.0           177775         22      8080.7       8152.5      1284      18221       4739.8  mmap          
      0.0           109053          1    109053.0     109053.0    109053     109053          0.0  pthread_join  
      0.0            78274          4     19568.5      13317.5     10422      41217      14570.3  write         
      0.0            61674         10      6167.4       3148.5      1483      26741       7791.1  fgets         
      0.0            23936          1     23936.0      23936.0     23936      23936          0.0  putc          
      0.0            22786          3      7595.3       3473.0      2468      16845       8026.2  fwrite        
      0.0            14252          4      3563.0       2750.0      1413       7339       2743.4  pipe2         
      0.0            11954          6      1992.3       1925.0      1144       2994        690.7  read          
      0.0             5529          3      1843.0       1298.0      1168       3063       1058.5  fcntl         
      0.0             2385          2      1192.5       1192.5      1107       1278        120.9  fflush        
      0.0             1218          1      1218.0       1218.0      1218       1218          0.0  close         

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)          Name        
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  --------------------
     89.3       1362349202          1  1362349202.0  1362349202.0  1362349202  1362349202          0.0  cudaEventSynchronize
      9.3        142073416          3    47357805.3      455766.0      243785   141373865   81420365.0  cudaMalloc          
      1.2         18745642          2     9372821.0     9372821.0     6176381    12569261    4520448.8  cudaMemcpy          
      0.1           802634          3      267544.7      207634.0      192270      402730     117325.7  cudaFree            
      0.0           495743          1      495743.0      495743.0      495743      495743          0.0  cudaLaunchKernel    
      0.0           456570          2      228285.0      228285.0        1182      455388     321172.1  cudaEventCreate     
      0.0           365509          2      182754.5      182754.5       10651      354858     243391.1  cudaEventRecord     
      0.0             5626          2        2813.0        2813.0         397        5229       3416.7  cudaEventDestroy    

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)                                              Name                                             
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ---------------------------------------------------------------------------------------------
    100.0       1362203062          1  1362203062.0  1362203062.0  1362203062  1362203062          0.0  void add_kernel_inner_loop<float>(T1 *, const T1 *, const T1 *, unsigned long, unsigned long)

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/cuda_gpu_mem_time_sum.py]... 

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation          
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
    100.0         12090479      2  6045239.5  6045239.5   6005240   6085239      56567.8  [CUDA memcpy Host-to-Device]

Processing [vector_add.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/reports/cuda_gpu_mem_size_sum.py]... 

 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
    134.218      2    67.109    67.109    67.109    67.109        0.000  [CUDA memcpy Host-to-Device]


```


# CUDA-GDB

## CUDA-GDB 支持 WSL2，但需要在 Windows 中把 EnableInterface 设置为 DWORD 1

powershell administration:

```
reg add "HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger" `
  /v EnableInterface `
  /t REG_DWORD `
  /d 1 `
  /f
```

重启电脑

## 编译运行

```
nvcc -g -G -O0 -std=c++17 vector_add.cu -o vector_add_debug

cuda-gdb ./vector_add_debug

```

WSL系统有点小兼容的问题
```
(base) chufeng@Chufeng:~/Desktop/InfiniTensor/coursedemo/cuda/01_vectorAdd_Profiling_GDB$ nvcc \
  -g \
  -G \
  -O0 \
  -std=c++17 \
  -arch=sm_86 \
  vector_add.cu \
  -o vector_add_debug
```

## 查询函数

```
(cuda-gdb) info functions add_kernel_inner_loop
```

```
(cuda-gdb) info functions add_kernel_inner_loop
All functions matching regular expression "add_kernel_inner_loop":

File /home/chufeng/Desktop/InfiniTensor/coursedemo/cuda/01_vectorAdd_Profiling_GDB/vector_add.cu:
72:     void add_kernel_inner_loop<float>(float*, float const*, float const*, unsigned long, unsigned long);

File /tmp/tmpxft_000040c1_00000000-6_vector_add.cudafe1.stub.c:
14:     static void __device_stub__Z21add_kernel_inner_loopIfEvPT_PKS0_S3_mm(float*, float const*, float const*, _ZSt6size_t, _ZSt6size_t);
15:     static void __wrapper__device_stub_add_kernel_inner_loop<float>(float*&, float const*&, float const*&, _ZSt6size_t&, _ZSt6size_t&);
(cuda-gdb) 
```

## 打断点

```
break vector_add.cu:72
```

## 运行

```
run

``` 

```
(cuda-gdb) run
Starting program: /home/chufeng/Desktop/InfiniTensor/coursedemo/cuda/01_vectorAdd_Profiling_GDB/vector_add_debug 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7fffe9f8d000 (LWP 6543)]
[New Thread 0x7fffe8b87000 (LWP 6544)]
[Detaching after fork from child process 6545]
[New Thread 0x7fffe3fff000 (LWP 6554)]
[Thread 0x7fffe3fff000 (LWP 6554) exited]
[New Thread 0x7fffe3fff000 (LWP 6555)]
[New Thread 0x7fffd56fe000 (LWP 6556)]
CUDA ELF Image contains unknown ABI version: 8
This might happen while debugging JITed codeusing latest driver with older tools.
Further debugging might not be reliable.Are you sure you want to continue? (y or [n]) y
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]

Thread 1 "vector_add_debu" hit Breakpoint 1, add_kernel_inner_loop<float><<<(1,1,1),(1,1,1)>>> (c=0x913c00000, a=0x90bc00000, b=0x90fc00000, n=16777216, step=1)
    at vector_add.cu:79
79          std::size_t idx =
```

```
(cuda-gdb) next
83          for (std::size_t i = idx; i < n; i += step) {
(cuda-gdb) print idx
$1 = 0
(cuda-gdb) next
84              c[i] = a[i] + b[i];
(cuda-gdb) print i
$2 = 0
(cuda-gdb) print a[i]
$3 = 1
(cuda-gdb) print c[i]
$4 = 0
(cuda-gdb) next
83          for (std::size_t i = idx; i < n; i += step) {
(cuda-gdb) print c[i]
$5 = 2
(cuda-gdb) 
```