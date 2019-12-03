# CUDA编程（一）

参考资料

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA编程(三): GPU架构了解一下!](https://www.jianshu.com/p/87cf95b1faa0)

CUDA C将C进行了拓展，其定义的C函数也称为``kernels``。``kernel``用关键字``__global__``来定义，调用``kernel``的形式为：``kernel_name<<<...>>>(paramlist)``，其中``kernel_name``是``kernel``函数的名字，而线程数、块数(``blocks``)则由``<<<...>>>``这一语法定义（称为**执行配置**语法），``(paramlist)``为输入``kernel``的参数。当``kernel``被调用执行时，该函数将同时被N个线程(``threads``)并行计算N次。每个线程都有一个独立的ID，可通过内置变量``threadIdx``来获得。譬如，以下程序定义了一个两向量相加的``kernel``，且执行时使用1个带有$N$线程的``block``：

```c
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int main()
{
	...
	// Kernel invocation with N threads
	VecAdd<<<1, N>>>(A, B, C);
	...
}
```

``<<<...>>>``的完整语法为``<<<Dg, Db, Ns, S>>>``，其中：

- ``Dg``可以是``int``型或者``dim3``型，指定了``block``的三个维度，``Dg.x*Dg.y*Dg.z``是总的block数量；
- ``Db``是``int``型或者``dim3``型，指定了每个``block``内的线程数量，``Db.x*Db.y*Db.z``；
- ``Ns``是``size_t``类型，指定了共享内存中的字节数量；可选，默认值为0；*待研究*
- ``S``是一个``cudaStream_t``类型的变量，指定特定数据流；可选，默认值为0；*待研究*

以下程序定义了一个两矩阵相加的``kernel``，且执行时使用了1个带有$N\times N$线程的``block``：

```c
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

## 内置关键字

- ``__device__``：定义了这样一个函数：
  1. 在设备上执行
  2. 只能从设备调用

- ``__global__``：定义了这样一个函数作为``kernel``：

  1. 在设备上执行
  2. 从宿主机调用
  3. 从计算力不小于3.2的设备调用（*待研究*）

  一个``__global__``函数必须是``void``类型的返回值，并且不能是类成员函数；

  任何对``__global__``函数的调用都必须指定**执行配置**；

  ``__global__``函数的调用都是异步的，可能在设备彻底完成任务前即返回值，所以可能造成CPU运行结束GPU仍在计算中或者没有获得``kernel``的返回值。为了处理这种情况，需要使用同步函数来使CPU进行等待直到GPU计算完成在继续后续的工作。常用的同步函数包括``cudaDeviceSynchronize``等。

- ``__host__``：定义了这样一个函数：

  1. 在宿主端执行
  2. 只能从宿主端调用

  相当于不加任何``CUDA``内置关键字的函数

``__device__``和``__global__``两者不能一起使用；``__global__``和``__host__``两者不能一起使用；而``__device__``和``__host__``两者可以一起使用，只是该函数将同时为宿主端和设备端进行编译。

## 内置变量

- ``gridDim``：``dim3``类型的变量，内容是``block``网格的维度
- ``blockIdx``：``uint3``类型的变量，内容是网格内某``block``的索引
- ``blockDim``：``dim3``类型，内容是某``block``内线程的维度
- ``threadIdx``：``uint3``类型的变量，内容是线程在``block``内的索引
- ``warpSize``：``int``类型（*待研究*）