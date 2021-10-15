# ModelZoo
Machine Learning models in Julia using [Flux.jl](https://fluxml.ai/Flux.jl/stable/)

Currently includes 
- a basic neural network which can classify handwritten digits (using the [MNIST](https://deepai.org/dataset/mnist) dataset)
- a convolutional neural network which can classify handwritten digits (again using the MNIST dataset)
- a convolutional neural network which can classify images (using the [CIFAR-10](https://deepai.org/dataset/cifar) dataset)

The convolutional neural networks use the [LeNet-5](https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342) architecture, first introduced by Yann LeCun et al. in 1998 in their paper [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf).

These models use [Pluto.jl](https://plutojl.org/), an interactive notebook similar to Jupyter.

For best results, install Pluto in the Julia REPL:

```using Pkg```
```Pkg.add("Pluto")```

Then open a new notebook with

```import Pluto```
```Pluto.run()```

and you can actually just paste the Github URL corresponding to the Julia (.jl) file you want to view.
(One of the benefits of Pluto is that packages are installed directly into the notebook, so there shouldn't be much more setup to do.)
