# A Low-Complexity MIMO Channel Estimator with Implicit Structure of a Convolutional Neural Network

This code was used to generate the figures in the article

B. Fesl, N. Turan, M. Koller, and W. Utschick, _A Low-Complexity MIMO Channel Estimator with Implicit Structure of a Convolutional Neural Network_, 2021, in press, arXiv pre-print: https://arxiv.org/abs/2104.12667. 

The paper is accepted for the 22nd IEEE International Workshop on Signal Processing Advances in Wireless Communications (SPAWC) 2021.

## Abstract
A convolutional neural network estimator which
learns the minimum mean squared error channel estimator for
single-antenna users was recently proposed. We generalize the
architecture to the estimation of MIMO channels with multiple antenna
users and incorporate assumptions, such as a high number
of antennas and a single propagation cluster, which reduces
the computational complexity tremendously. Learning is used in
this context to combat the mismatch between the assumptions
and real scenarios with a limited number of antennas and many
propagation clusters. We derive a high-level description of the
estimator for arbitrary choices of the pilot sequence. It turns
out that the proposed estimator has the implicit structure of
a two-layered convolutional neural network, where the derived
quantities can be relaxed to learnable parameters. We show that
by using discrete Fourier transform based pilots the number of
network parameters decreases significantly and the online run
time of the estimator is reduced considerably, where we can
achieve linearithmic order of complexity in the number of antennas.
Numerical results display performance gains compared to
state-of-the-art algorithms from the field of compressive sensing
or covariance estimation of the same or higher computational
complexity. The simulation code is available online.

## Installation Notes
This code is written in _Python_ (https://www.python.org) and uses _TensorFlow_ together with _Keras_ as backend.
The code was tested with _Python_ version 3.7, _TensorFlow_ version 2.2.0.
## License
This code is licensed under 3-clause BSD License:

>Copyright (c) 2021 B. Fesl, N. Turan, and M. Koller.
>
>Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
>
>1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
>
>2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
>
>3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
>
>THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
