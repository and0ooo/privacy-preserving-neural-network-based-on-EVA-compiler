# Privacy-Preserving Neural Network Based on EVA Compiler

An implementation of privacy-preserving neural network (cryptonets and squeezenet) based on EVA compiler. ("EVA: An Encrypted Vector Arithmetic Language and Compiler for Efficient Homomorphic Computation" Dathathri et al.)

## Environment

1. Pytorch.
2. EVA compiler. Reference: https://github.com/microsoft/EVA

## Usage

1. `python main.py` to train cryptonets or squeezenet in plaintext. 
2. `python parameter.py` to store parameters of cryptonets or squeezenet.
3. `python cryptonets_ciphertext.py` or `python squeezenet_ciphertext.py` to inference in Ciphertext.

Parameters:


```
--save 
    Save to files or not. Default is false.

--read
    Read from files or not. Default is true.

--path
    If '--save' is true, 'path' defines the saving path.

--img
    Path of img.

--param
    Path of parameters.
```


## Cryptonets

Dataset: MNIST.


Network structure:

```
Conv2d;
    - input channel: 1; output channel: 3;
    - kernel size: 5; stride: 2; padding: 1; bias: False;

Quadratic polynomial activation;

AvgPool2d;
    - kernel size: 3; stride: 1; padding: 1;

Conv2d;
    - input channel: 3; output channel: 5;
    - kernel size: 5; stride: 2; padding: 0; bias: False;

AvgPool2d;
    - kernel size: 3; stride: 1; padding: 1;

Linear;
    - input: 125; output: 10.

LogSoftmax.
```

Experiment result:

```
plaintext inference accuracy: 97%
ciphertext inference accuracy: 97%
key generation: 239.55s
encryption: 0.0776s
execution: 170.5s
decryption: 0.0126s
memory consumption: 160G
```

Due to some redundant operations in the ciphertext inference stage, the cryptonets ciphertext inference delay is longer than the results in the paper.

## Squeezenet

Dataset: CIFAR10.

Network structure:

```
Conv2d;
    - input channel: 3; output channel: 3;
    - kernel size: 3; stride: 1; padding: 1; bias: False;
Quadratic polynomial activation;
BatchNorm2d;
AvgPool2d;
    - kernel size: 3; stride: 2; padding: 1;

fire;
    - inplanes: 3; squeeze_planes:12; expand_planes:8;
fire;
    - inplanes: 16; squeeze_planes:12; expand_planes:6;
AvgPool2d;
    - kernel size: 3; stride: 2; padding: 1;

Conv2d;
    - input channel: 12; output channel: 40;
    - kernel size: 3; stride: 1; padding: 1; bias: False;
Quadratic polynomial activation;
BatchNorm2d;

Conv2d;
    - input channel: 40; output channel: 40;
    - kernel size: 3; stride: 1; padding: 1; bias: False;
Quadratic polynomial activation;
BatchNorm2d;

Conv2d;
    - input channel: 40; output channel: 10;
    - kernel size: 1; stride: 1; padding: 0; bias: False;
Quadratic polynomial activation;
BatchNorm2d;

AvgPool2d;
    - kernel size: 8; stride: 8; padding: 0;

LogSoftmax.
```

Fire module structure:

```
Conv2d
    - input channel: inplanes; output channel: squeeze_planes;
    - kernel size: 1; stride: 1; padding: 0; bias: False;
Quadratic polynomial activation;
BatchNorm2d;

Conv2d1
    - input channel: squeeze_planes; output channel: expand_planes;
    - kernel size: 1; stride: 1; padding: 0; bias: False;
Quadratic polynomial activation;
BatchNorm2d;

Conv2d2
    - input channel: squeeze_planes; output channel: expand_planes;
    - kernel size: 3; stride: 1; padding: 1; bias: False;
Quadratic polynomial activation;
BatchNorm2d;

concatenate[Conv2d1, Conv2d2];
```

Squeezenet cannot run on ciphertext because it still has many redundant operations. Lowering the SLOTS value is one way to reduce operations.

```python
                                    # slots     # 4096            # 8192 less     # 8192          # 16384
CONV1_FILTER_NUM =                  3           # 3               # 7             # 7             # 14
FIRE2_SQUEEZE_PLAINES =             12          # 12              # 20            # 25            # 32
FIRE2_EXPAND_PLAINES =              8           # 8               # 16            # 16            # 32
FIRE3_SQUEEZE_PLAINES =             12          # 12              # 20            # 25            # 32
FIRE3_EXPAND_PLAINES =              6           # 6               # 12            # 12            # 25
CONV2_FILTER_NUM =                  40          # 40              # 20            # 81            # 160
CONV3_FILTER_NUM =                  40          # 40              # 20            # 81            # 160
```

## Interaction

This repository also contains code for client and server interaction. `python client.py` and `python server.py` to run.


## Reference

1. training implementation: 
   1. https://github.com/gsp-27/pytorch_Squeezenet
2. quadratic polynomial activation: 
   1. https://blog.csdn.net/fupotui7870/article/details/115732660
3. cryptonets structure: 
   1. https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/CryptonetsTechReport.pdf
4. squeezenet structure:
   1. reference 1 (HEMET): https://arxiv.org/pdf/2106.00038.pdf
   2. reference 2 (EVA): https://arxiv.org/pdf/1912.11951.pdf
   3. reference 3: https://github.com/kaizouman/tensorsandbox/tree/master/cifar10/models/squeeze
5. conv:
   1. reference 1 (HEMET): https://arxiv.org/pdf/2106.00038.pdf
   2. reference 2: https://github.com/microsoft/EVA/blob/main/tests/large_programs.py
   3. reference 3: https://github.com/microsoft/EVA/blob/main/examples/image_processing.py
6. horizontal sum: 
   1. https://github.com/microsoft/EVA/issues/6#issuecomment-761295103
7. interaction: 
   1. https://github.com/Chen-Junbao/SecureAggregation/blob/master/utils.py
   2. https://github.com/microsoft/EVA/blob/main/examples/serialization.py
