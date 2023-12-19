# NOTE !
# my code submission is under /home/zc2645/conformer_libtorch

# since i don't use kaldi, I did not put the file in kaldi foler

# conformer_libtorch

a libtorch version of conformer encoder with a simple decoder

strictly following the paper of Conformer

(this is model implementation, training is currently not supported, I extracted the tensor for future use with preprocess.py)

## Instruction
sh install.sh

sh run.sh

./build/test

sh prepare.sh (optional for data prepare and preprocess)

## 

# Reference
Conformer paper

preprocess all follows the instruction in the paper.

https://github.com/msalhab96/Conformer

https://github.com/k2-fsa/icefall


## NOTE
this is not complete, I am not able to get a full training. Still too challanging for me. 

All the code can be compiled !

All module test passed ! (test with dimension check)

I am currently not able to get a full training in libtorch.

Libtorch still have some issue

The register_buffer has device issue that I cannot deal with


