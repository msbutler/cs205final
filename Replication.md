## Replication Instructions

1. Spin up [AWS Deep Learning AMI (Ubuntu 18.04 OS) Version 43.0](https://aws.amazon.com/marketplace/pp/B07Y43P7X5). To access a gpu, use a g3.4x large infrastructure, and to replicate our performance results, use a g3.16xlarge.

2. Type the following in your terminal (assumes your working ssh key for aws is `~/.ssh/CS205-key.pm`)
`chmod 0400 ~/.ssh/CS205-key.pem` 

3. Type into terminal (maybe use the public dns or public ip....whatever works)
`ssh -L localhost:8888:localhost:8888 -i ~/.ssh/CS205-key.pem ubuntu@<Your public instance DNS>`

4. Next steps are on the machine. Pull the git repo:
`git clone https://github.com/msbutler/cs205final`

5. Activate the proper environment:
`source activate tensorflow_p37`

6. Run the vanilla model (within the file, set semi=True for the semisupervised model and False for the fully supervised version)
`python run.py`

7. Run the performance tests
   - Weak Scaling: Run `python ./weak_scaling.py` in the terminal to increase the problem size while keeping the computing power fixed. Change the semi input to `True` for the semisupervised model or `False` for the fully supervised model. This code will produce a plot of the average training time per epoch in the `Figures` directory.  
   - Strong Scaling: Run `python ./strong_scaling.py [True/False] [#GPUs]` in the terminal to increase the computing power while maintaining the problem size. The first input dictates which model to use (`True` for semisupervised and `False` for fully supervised) while the second input determines the number of GPUs to be used. This code prints the average training time per epoch to the terminal.

Instructions loosely base based on [this](https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/).

## Additional Replication Information:

NVIDIA-SMI 450.119.03   Driver Version: 450.119.03   CUDA Version: 11.0 

Architecture:        x86\_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              32
On-line CPU(s) list: 0-31
Thread(s) per core:  2
Core(s) per socket:  16
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               79
Model name:          Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
Stepping:            1
CPU MHz:             1689.978
CPU max MHz:         3000.0000
CPU min MHz:         1200.0000
BogoMIPS:            4600.00
Hypervisor vendor:   Xen
Virtualization type: full
L1d cache:           32K
L1i cache:           32K
L2 cache:            256K
L3 cache:            46080K
NUMA node0 CPU(s):   0-31
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx pdpe1gb rdtscp lm constant\_tsc rep\_good nopl xtopology nonstop\_tsc cpuid aperfmperf pni pclmulqdq ssse3 fma cx16 pcid sse4\_1 sse4\_2 x2apic movbe popcnt tsc\_deadline\_timer aes xsave avx f16c rdrand hypervisor lahf\_lm abm 3dnowprefetch cpuid\_fault invpcid\_single pti fsgsbase bmi1 hle avx2 smep bmi2 erms invpcid rtm rdseed adx xsaveopt
