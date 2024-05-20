nvcc -diag-suppress 550 main.cu -o main -Xcompiler /openmp
 
@del main.lib
@del main.exp
@del _CL*