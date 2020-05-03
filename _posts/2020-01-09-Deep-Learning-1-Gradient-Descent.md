---
title: "Deep Neural Network 1: Gradient Descent and Logistic Regression"
last_modified: 2020-01-09
mathjax: true
categories: 
  - blogs
  - deep-learning
---

Gradient descent is an important technique in machine learning to optimize the cost function. In this article, we will first go over the high-level ideas of the gradient-descent, and then look at an example on its application to a simple logistic regression model.

## Gradient Descent

<p align="center">
  <img src="/assets/dl_posts/convex_function.png" width="500" height="500" title="Figure 1. Convex function">
</p>


Let us start with a convex function $$y = f(x)$$, as shown in Figure 1. It has one minimum point and we can obtain the optimal $$x$$ by solving $$\frac{dy}{dx} = 0$$. However, it would become difficult when there are more parameters to the function. Instead, we can use an iterative approach to obtain the optimal or suboptimal $$x$$:  
1. Start with a random x value.  
2. Calculate the derivative $$\frac{df(x)}{dx}$$.  
3. Move to the direction where $$f(x)$$ is decreasing with the step size that is proportional to the derivative.  
4. Repeat 1) ~ 3) until the derivative is close or equal to 0.  


The above process can be simplified as:  

**Repeat:**
$$ x := x - \alpha \frac{df(x)}{dx}.$$ until $$x$$ converges.

<p align="center">
  <img src="/assets/dl_posts/gradient_descent_good.png" width="500" height="500" title="Figure 2. Gradient Descent">
</p>

Thus, with a proper value of $$\alpha$$, x moves toward the point that minimizes $$f(x)$$, as shown in Figure 2. It can be easily seen that $$\alpha$$ controls the pace of the iteration toward the minimum point. In machine learning problems, this in general determines how fast the trained model approaches/learns the real model from data, so we call it learning rate. However, we don't want the learning rate to be either too big or too small: too big learning rate can bypass the minimum point and cause osillaction, while too small learning rate may take long time to converge or stuck in some local optimal.


## Logistic Regression
In machine learning, Logistic Regression is normally used for classification problems, such as number classification or dog detector. Let us consider a simple model with two parameters with input $$(x_1, x_2)$$ and output $$y$$:  

$$ 
z = w_1 x_1 + w_2 x_2 + b \\
a = \sigma(z), \\
$$

where $$\sigma$$ is the activation function such as sigmoid function. Thus, the loss function is:

$$ \mathcal{L}(a, y) = a - y. $$

where $$a$$ is the prediction result from the model. The goal is to find the parameter set $$(w_1, w_2, b)$$ that minimizes $$\mathcal{L}$$, or:

$$ \operatorname*{argmin}_{w_1, w_2, b} \mathcal{L}(a, y) $$

But how do we get that?

## Apply Gradient Descent
With the idea of gradient descent, if we know the gradient of the loss function $$\mathcal{L}$$ along the $w_1, w_2, b$, we may find the directions that decreases the prediction error (in a high dimenstional graph representation). Thus, we need to get the derivatives of the loss function to each parameter. We simplify the expressions as:

$$
\frac{\partial \mathcal{L}}{\partial z} \rightarrow d z,  
\frac{\partial \mathcal{L}}{\partial w_1} \rightarrow d w_1,  
\frac{\partial \mathcal{L}}{\partial w_2} \rightarrow d w_2,  
\frac{\partial \mathcal{L}}{\partial b} \rightarrow d b.
$$

with

$$
dz = a - y, \\
d w_1 = dz \cdot \frac{dz}{d w_1} = x_1 \cdot dz, \\
d w_2 = dz \cdot \frac{dz}{d w_2} = x_2 \cdot dz, \\
d b = dz \cdot \frac{dz}{db} = dz. 
$$

and we are happy to get the gradient descent function that can bring us to the happy ending:

$$
w_1 := w_1 - \alpha \cdot d w_1 \\
w_2 := w_2 - \alpha \cdot d w_2 \\
b := b - \alpha \cdot d b
$$

However, this would not work well if we only have one sample as these parameters will change with constant rates and never converge. 

## Multiple Data Points
Let us assume there are $$m$$ data points. There will be $$m$$ prediction errors. By averaging over the prediction errors, we can obtain the cost function we want to optimize against:

$$ J(\textbf{w}, b) = \frac{1}{m} \sum_{i=1}^m \mathcal{L} (a^{(i)}, y^{(i)}), $$

where $$\textbf{w} = [w_1, w_2]$$, and the superscript represents the result for the $$i$$th sample. Conceptually, for each step, we would get direction from each data point, and we need to get the average one to update:

$$
dz^{(i)} = a^{(i)} - y^{(i)}, \\
d \textbf{w} = \frac{1}{m} \sum_{i=1}^m d \textbf{w}^{(i)} = \frac{1}{m} \sum_{i=1}^m \textbf{x}^{(i)} \cdot dz^{(i)}, \\
d b = \frac{1}{m} \sum_{i=1}^m d b^{(i)} = \frac{1}{m} \sum_{i=1}^m dz^{(i)}.
$$

## Vectorization
In order to obtain $d w_1, d w_2, d b$ for m examples in each epoch, we need to implement for loop to go through each sample. However, this is not efficient - there are two for loops: one for epochs, and one for data samples iterations. The good news is that we can use matrix operation to compute for all data samples at the same time.

Let the input samples as matrix $$\textbf{X} = [\textbf{x}^{(1)}, ..., \textbf{x}^{(m)}]$$, where $$i$$th sample is $$\textbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]^T$$, and the output samples as $$\textbf{Y} = [y_1, y_2, ..., y_m]$$. The output would be:

$$
\textbf{Z} = \textbf{w} \textbf{X} + \textbf{b} \\
\textbf{A} = \sigma(\textbf{Z}) \\
$$

Thus, we can futher obtain the average gradients with matrix operations:

$$
d\textbf{Z} = \textbf{A} - \textbf{Y} \\
d\textbf{w} = \frac{1}{m} \textbf{X} \cdot d\textbf{Z}^T \\
d\textbf{b} = \frac{1}{m} np.sum(d \textbf{Z})
$$

Finally, we get one step to walk with. So how about next? We will repeat this process with a few more steps until it is enough. So when is enough? Well, there are usually two conditions we can use for judgement:
1. A pre-defined number of steps has been reached. 
2. The parameters converge, which means the step sizes are below certain thresholds for a few steps.

We can stop when either of the condition has been met and get our trained models.

## Summary
From the above example, we define the model for the problem, compute the cost function and apply the gradient descent to optimize model parameters. The vectorization technique to improve the computation efficiency is also discussed. These are also the techniques we would use in a practical ML problem. Based on the model performance, we could use other techniques to further improve our models. 

