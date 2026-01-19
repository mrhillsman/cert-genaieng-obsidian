## Gradient Descent

![[videoframe_46889.png]]

![[videoframe_97953.png]]

*gradient descent* => iterative optimization algorithm for finding the minimum of a function

![[videoframe_193045.png]]

![[videoframe_275895.png]]


How do we determine the best value of w?
- start with a random value for w - 0.2
- to determine which direction to move compute the gradient of the lost function at the current value of w
- gradient is given by the slope of the tangent at w = 0.2 and the magnitude of the step is determined by the learning rate

w1 -> w0 - n (learning rate) times delta J over delta w (slope of the tangent at w=0.2)



# Gradient Descent - Student Notes

## Introduction to Gradient Descent 📉

_As a beginner in machine learning, this topic looks important for understanding how models actually "learn"_

Gradient descent is an **iterative optimization algorithm** used for finding the minimum of a function. It's a key algorithm used to optimize weights and biases in neural networks and other machine learning models.

> 🧠 **Core concept**: We're trying to find the values of parameters that minimize a cost function by taking steps in the direction of steepest descent.

## Cost Functions 📊

Before understanding gradient descent, I need to understand what a cost function is:

- A cost function (also called loss function) measures how poorly our model is performing
- It quantifies the difference between predicted values and actual values
- The goal is to **minimize** this function

In the example from the lecture:

- We have data points where $z = 2x$ (as shown in Image 4)
- We want to find the weight $w$ that best fits this relationship

The cost function used is:

$$\Large{J(w) = \frac{1}{2m}\sum_{i=1}^{m}(z_i - wx_i)^2}$$

Where:

- $m$ is the number of data points
- $z_i$ is the actual value
- $wx_i$ is the predicted value
- We square the differences to ensure positive values and penalize larger errors more

> 📝 **Note**: This is actually the Mean Squared Error (MSE) function divided by 2. I should research why we divide by 2 (probably to make derivatives cleaner).
> 
> Yes. Including the factor of $\large{\frac{1}{2}}$ in the mean squared error (MSE) cost function effectively cancels out the 2 that appears when taking the derivative of $(z_i - w,x_i)^2$, making the gradient simpler.
> 
> Using First Principles Thinking:
> 
> 1. **Start with the original MSE**: $\large{\text{MSE} = \frac{1}{m}\sum_{i=1}^{m}(z_i - w,x_i)^2}$
> 2. **Observe the derivative**: its gradient with respect to $w$ introduces a factor of 2 from the power of 2 in the squared term.
> 3. **Introduce the factor $\large{\frac{1}{2}}$**: $\large{\frac{1}{2m}\sum (z_i - w,x_i)^2}$ so that when you differentiate, the $\large{2}$ cancels with $\large{\frac{1}{2}}$, leaving a simpler form.
> 4. **Result**: This does not affect the location of the minimum—just makes the calculus cleaner by avoiding extra constants in the gradient.

## What Makes This Cost Function Useful? 🤔

From the lecture and Image 3:

- This cost function forms a **parabola** with one global minimum
- This means there's a single, unique solution (no local minima to get stuck in)
- For our simple dataset, the optimal value is $w = 2$ (matches the actual relationship!)

> ❓ **Question**: What happens when cost functions have multiple local minima? Need to research this more.
> 
> When a cost function has multiple local minima, the optimization process can get stuck in one of these “valleys” rather than the absolute lowest point (“global minimum”). In practical terms, an algorithm such as gradient descent may converge to different local minima depending on the starting conditions and the shape of the function, potentially resulting in suboptimal solutions.
> 
> **Storytelling Technique**:  
> Imagine you’re hiking through a mountain range filled with many valleys. Each valley floor is a local minimum. If your method of finding the lowest point is just walking downhill from your current position, you’ll end up in the nearest valley, not necessarily the deepest one across the entire range. The presence of multiple valleys (local minima) means you can get stuck in one of them before reaching the global lowest point.

## How Gradient Descent Works 🔍

Looking at Image 2, gradient descent works like this:

1. **Initialize**: Start with a random value for weight $w$ (e.g., $w_0 = 0.2$)
2. **Compute gradient**: Calculate the slope of the cost function at current point
3. **Update weight**: Move in the opposite direction of the gradient (downhill)
4. **Repeat**: Keep updating until we reach the minimum (or close enough)

The weight update formula is:

$$\Large{w^{t+1} = w^t - \eta \frac{\partial J}{\partial w}}$$

Where:

- $w^t$ is the current weight value
- $w^{t+1}$ is the updated weight value
- $\eta$ (eta) is the learning rate
- $\frac{\partial J}{\partial w}$ is the gradient (slope) of the cost function

> 📝 **TODO**: Find out how to actually calculate the gradient $\frac{\partial J}{\partial w}$ for this specific cost function.

## Learning Rate Considerations ⚙️

The learning rate $\eta$ controls how big of a step we take with each iteration:

- **Too large**: We might overshoot the minimum and fail to converge
- **Too small**: Algorithm will take too long to reach the minimum

> 🔍 **Research needed**: How to choose appropriate learning rates for different problems?
> 
> Here are practical heuristics:
> 
> **1. The 5 Whys**
> 
> - **Why** do we need a specific range? Because each problem’s loss landscape has different curvature.
> - **Why** does curvature matter? Steep slopes demand smaller steps to avoid overshooting.
> - **Why** do we overshoot? A large learning rate multiplies the gradient too much.
> - **Why** not always choose a tiny rate? Training time would explode.
> - **Why** not always choose a large rate? It might never converge.
> 
> **2. The 80/20 Rule (Pareto Principle)**
> 
> - A small fraction of hyperparameter tuning steps can yield most of the benefits. Start with a broad search (e.g., 0.1 down to 1e-6) in powers of 10, identify the promising range, then fine-tune within that.
> 
> **3. Ladder of Inference**
> 
> 1. Gather data: run small experiments with different rates.
> 2. Observe patterns: note training/validation loss behavior (oscillation vs. slow convergence).
> 3. Draw conclusions: narrow to the range where training is stable yet not too slow.
> 4. Refine: do more granular searches (like 0.001, 0.002, 0.005).
> 
> **In Formula Terms**  
> If the model updates via gradient descent with step size $\large{\alpha}$,
> 
> $\large{w \leftarrow w - \alpha\,\nabla_w \, J(w)}$ your goal is to choose $\large{\alpha}$ so training is both stable and efficient. You typically determine this by trial, error, and observing the loss curve.
> 
> **Key Takeaways**
> - Begin with a wide logarithmic sweep for $\large{\alpha}$ (e.g., 1e-3, 1e-2, 1e-1) - Narrow down once you see how the model responds
> - Look for the “sweet spot” where the loss decreases smoothly without oscillation or stagnation

## Visualizing the Iterations 👁️

From Image 1, we can see the progression of gradient descent over 4 iterations:

|Iteration|Weight Value|Line Fit|Comment|
|---|---|---|---|
|Initial|$w = 0.000$|Horizontal line|Very poor fit, high cost|
|1|$w = 1.105$|Positive slope|Big improvement, steep gradient led to big step|
|2|$w = 1.600$|Steeper slope|Better fit, gradient not as steep so smaller step|
|3|$w = 1.821$|Getting closer|Even better fit|
|4|$w = 1.920$|Almost there|Nearly optimal fit|

> 💡 **Insight**: Notice how the steps get smaller as we get closer to the minimum! This is because the gradient gets smaller as we approach the minimum.

## Direction of Movement 🧭

An important point from the lecture:

- If we initialize $w$ to a value **to the left** of minimum: positive gradient → move right
- If we initialize $w$ to a value **to the right** of minimum: negative gradient → move left
- The algorithm always moves toward the minimum regardless of starting point

## Extending to Real-World Applications 🌐

In practice, gradient descent gets more complex:

- Multiple parameters to optimize (not just one $w$)
- Can't visualize high-dimensional cost functions
- May encounter non-convex cost functions with multiple local minima

> 📚 **Research topics**:
> 
> - Stochastic Gradient Descent (SGD)
> - Mini-batch Gradient Descent
> - Momentum, AdaGrad, RMSProp, Adam optimizers

## My Questions for Further Study 🤔

1. How do we calculate the gradient for more complex models with multiple parameters?
2. What techniques exist to avoid getting stuck in local minima?
3. How is gradient descent applied in neural networks specifically?
4. What's the mathematical derivation of the gradient for the MSE cost function?
5. How do we know when to stop the algorithm (convergence criteria)?

## Practical Python Implementation (To Try Later) 💻

Here's what I think the code might look like for implementing simple gradient descent:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data (z = 2x)
x = np.linspace(-1, 1, 20)
z = 2 * x

# Initialize weight and learning rate
w = 0
learning_rate = 0.4
iterations = 5
cost_history = []

# Gradient descent algorithm
for i in range(iterations):
    # Predictions
    z_pred = w * x
    
    # Calculate cost (MSE)
    cost = (1/(2*len(x))) * np.sum((z - z_pred)**2)
    cost_history.append(cost)
    
    # Calculate gradient
    gradient = -(1/len(x)) * np.sum((z - z_pred) * x)
    
    # Update weight
    w = w - learning_rate * gradient
    
    print(f"Iteration {i}: w = {w:.4f}, Cost = {cost:.4f}")

# Plot final fit
plt.scatter(x, z)
plt.plot(x, w * x, color='red')
plt.title(f'Final fit: w = {w:.4f}')
plt.xlabel('x')
plt.ylabel('z')
plt.show()
```

> 🔍 **TODO**: Test this code and verify if it's correct! Need to make sure my gradient calculation is right.

## Summary of Key Points 📝

- **Gradient descent** is an iterative algorithm to find the minimum of a function
- It works by taking steps proportional to the negative of the gradient
- The **learning rate** controls the step size and needs careful tuning
- For simple linear models with MSE cost functions, we get a nice parabola with one global minimum
- The algorithm converges when we reach (or get very close to) the minimum value

## Glossary 📖

- **Cost/Loss Function**: Measures how poorly our model is performing
- **Gradient**: The slope or derivative of the cost function with respect to the parameters
- **Learning Rate**: Parameter that controls the size of steps during gradient descent
- **Convex Function**: A function with only one minimum (like our parabola)
- **Global Minimum**: The lowest point of the entire function
- **Local Minimum**: A low point in a region of the function, but not necessarily the lowest overall
- **Convergence**: When the algorithm reaches (or gets very close to) the minimum

> 🌟 **Final note**: Gradient descent seems fundamental to understanding how neural networks and many other ML algorithms work. I definitely need to understand this well before moving on to more complex topics!


___

## Backpropagation

# Backpropagation - Student Notes

## Introduction 🧠

These notes cover backpropagation, the algorithm used to train neural networks. As a beginner to ML, this is a crucial concept that connects to the gradient descent algorithm we studied earlier.

> 💡 **Key concept**: Backpropagation is how neural networks learn from their mistakes by propagating error backwards through the network to update weights and biases.

## Neural Network Training Overview

In previous material, we learned about forward propagation (how neural networks make predictions). But how do networks learn to optimize their weights and biases?

Training happens in a **supervised learning** setting where:
- Each data point has a corresponding label or ground truth
- Training is needed when predictions don't match ground truth
- The goal is to minimize the error between predictions and actual values

## The Backpropagation Process 📊

From Image 1 (Complete Training Algorithm) and Image 15, the overall process is:

1. Initialize weights and biases to random values
2. Iteratively repeat:
    - Calculate network output using forward propagation
    - Calculate error between ground truth and predicted output
    - Update weights and biases through backpropagation
    - Repeat until convergence (predefined iterations/epochs or error below threshold)

## Error Calculation 📉

The error (cost/loss function) for a simple network is calculated as:

$$\Large{E = \frac{1}{2}(T - a_2)^2}$$

Where:

- $T$ is the ground truth (target)
- $a_2$ is the predicted output
- For multiple data points, the mean squared error is used: $$\Large{E = \frac{1}{2m}\sum_{i=1}^{m}(T_i - a_{2,i})^2}$$

> 📝 **Note**: This is similar to the cost function from gradient descent, but now applied to neural network outputs.

## Simple Network Example

From Images 6-7, our example has:
- A simple network with 2 neurons
- Input $x_1 = 0.1$
- Initial weights: $w_1 = 0.15$, $w_2 = 0.45$
- Initial biases: $b_1 = 0.40$, $b_2 = 0.65$
- Ground truth $T = 0.25$
- Learning rate $\eta = 0.4$
- 1000 epochs (maximum iterations)
- Error threshold $\epsilon = 0.001$

### Forward Propagation Calculations:

1. First layer:
    - $\large{z_1 = w_1 \cdot x_1 + b_1 = 0.15 \cdot 0.1 + 0.4 = 0.415}$
    - $\large{a_1 = f(z_1) = \frac{1}{1+e^{-z_1}} = \frac{1}{1+e^{-0.415}} = 0.6023}$
2. Second layer:
    - $\large{z_2 = a_1 \cdot w_2 + b_2 = 0.6023 \cdot 0.45 + 0.65 = 0.9210}$
    - $\large{a_2 = f(z_2) = \frac{1}{1+e^{-z_2}} = \frac{1}{1+e^{-0.9210}} = 0.7153}$
3. Error calculation:
    - $\large{E = \frac{1}{2}(T - a_2)^2 = \frac{1}{2}(0.25 - 0.7153)^2 = 0.1083}$

> ❓ **Question to investigate**: Why use sigmoid activation? What are its properties and limitations?

## Updating Weights and Biases with Backpropagation

The key idea: we need to calculate $\frac{\partial E}{\partial w}$ for each weight and bias, then update using:

$$\Large{w \rightarrow w - \eta \cdot \frac{\partial E}{\partial w}}$$ $$\Large{b \rightarrow b - \eta \cdot \frac{\partial E}{\partial b}}$$

The challenge is that the error isn't directly a function of weights/biases, so we need the **chain rule**.

### 1. Updating $w_2$ (Image 5, 12-13)

Since $E$ depends on $a_2$, which depends on $z_2$, which depends on $w_2$:

$$\Large{\frac{\partial E}{\partial w_2} = \frac{\partial E}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial w_2}}$$

Where:

- $\frac{\partial E}{\partial a_2} = -(T - a_2)$
- $\frac{\partial a_2}{\partial z_2} = a_2(1 - a_2)$ (derivative of sigmoid)
- $\frac{\partial z_2}{\partial w_2} = a_1$

So: $$\Large{\frac{\partial E}{\partial w_2} = -(T - a_2) \cdot a_2(1 - a_2) \cdot a_1}$$

Plugging in the values: $$\Large{\frac{\partial E}{\partial w_2} = -(0.25 - 0.7153) \cdot 0.7153(1 - 0.7153) \cdot 0.6023 = 0.05706}$$

New $\large{w_2 = 0.45 - 0.4 \cdot 0.05706 = 0.427}$

### 2. Updating $b_2$ (Image 4, 11)

Similarly: $$\Large{\frac{\partial E}{\partial b_2} = \frac{\partial E}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial b_2}}$$

Where:

$$\Large{\frac{\partial z_2}{\partial b_2} = 1}$$

So: $$\Large{\frac{\partial E}{\partial b_2} = -(T - a_2) \cdot a_2(1 - a_2) \cdot 1}$$

Plugging in the values: $$\Large{\frac{\partial E}{\partial b_2} = -(0.25 - 0.7153) \cdot 0.7153(1 - 0.7153) \cdot 1 = 0.0948}$$

New $\large{b_2 = 0.65 - 0.4 \cdot 0.0948 = 0.612}$

### 3. Updating $w_1$ (Image 3, 9-10)

This is trickier because we need to go back further in the network:

$$\Large{\frac{\partial E}{\partial w_1} = \frac{\partial E}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}}$$

Where:

- $\frac{\partial z_2}{\partial a_1} = w_2$
- $\frac{\partial a_1}{\partial z_1} = a_1(1 - a_1)$ (derivative of sigmoid)
- $\frac{\partial z_1}{\partial w_1} = x_1$

So: $$\Large{\frac{\partial E}{\partial w_1} = -(T - a_2) \cdot a_2(1 - a_2) \cdot w_2 \cdot a_1(1 - a_1) \cdot x_1}$$

Plugging in: $$\Large{\frac{\partial E}{\partial w_1} = -(0.25 - 0.7153) \cdot 0.7153(1 - 0.7153) \cdot 0.45 \cdot 0.6023(1 - 0.6023) \cdot 0.1 = 0.001021}$$

New $\large{w_1 = 0.15 - 0.4 \cdot 0.001021 = 0.1496}$

### 4. Updating $b_1$ (Image 2, 8)

Similar to $w_1$:

$$\Large{\frac{\partial E}{\partial b_1} = \frac{\partial E}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial b_1}}$$

Where:

$$\Large{\frac{\partial z_1}{\partial b_1} = 1}$$

So: $$\Large{\frac{\partial E}{\partial b_1} = -(T - a_2) \cdot a_2(1 - a_2) \cdot w_2 \cdot a_1(1 - a_1) \cdot 1}$$

Plugging in: $$\Large{\frac{\partial E}{\partial b_1} = -(0.25 - 0.7153) \cdot 0.7153(1 - 0.7153) \cdot 0.45 \cdot 0.6023(1 - 0.6023) \cdot 1 = 0.01021}$$

New $\large{b_1 = 0.40 - 0.4 \cdot 0.01021 = 0.3959}$

> 📝 **Important**: The PDF document shows detailed derivations of these partial derivatives, focusing on calculus steps I need to review.

## Chain Rule in Backpropagation 🔄

The chain rule is essential to backpropagation! I'm noting the pattern:

1. For output layer weights/biases ($w_2$, $b_2$):
    - Only need to go back one step
    - Derivatives are simpler
2. For hidden layer weights/biases ($w_1$, $b_1$):
    - Need to propagate error through multiple layers
    - Derivatives involve more terms

Image 10 shows this clearly for $w_1$:

- Start with error $E$
- Follow connections: $a_2 \rightarrow z_2 \rightarrow a_1 \rightarrow z_1 \rightarrow w_1$
- Calculate derivatives at each step
- Multiply them all together

## The Complete Training Process 🔄

After the first iteration:

1. We have updated weights and biases
2. We do another round of forward propagation with the new parameters
3. Calculate the new error
4. Do another round of backpropagation
5. Continue until convergence

## Mathematical Insights from Derivatives 🧮

### For Sigmoid Function $f(z) = \frac{1}{1+e^{-z}}$:

The derivative is: $$\Large{\frac{df(z)}{dz} = f(z)(1-f(z))}$$
This appears repeatedly in our calculations and is why sigmoid functions were historically popular (easy derivatives).

### Error Gradient:

For any parameter θ, the gradient has form: $$\Large{\frac{\partial E}{\partial \theta} = -(T - a_2) \cdot (\text{chain of derivatives})}$$
The term $-(T - a_2)$ represents the direction to move - if prediction $a_2$ is too high compared to target $T$, we'll decrease the weights.

## Limitations and Questions ❓

The lecture mentioned a "serious shortcoming" of sigmoid functions in deep networks that will be discussed next.

> 🔍 **TODO**: Research "vanishing gradient problem" - might be related to sigmoid limitations

## Personal Notes/Questions 📝

1. The chain rule is crucial for backpropagation - I need to review calculus!
2. Visualization: error flows backward through the network
3. Each weight update depends on:
    - How much the network is wrong (error term)
    - How much that weight contributes to the error
    - The learning rate
4. **Questions to investigate**:
    - How does this scale to deep networks with many layers?
    - How are matrices used to make this more efficient?
    - What are alternatives to sigmoid activation functions?
    - How does regularization fit into backpropagation?
    - What exactly is the "vanishing gradient problem"?

## Python Implementation Sketch (to try later) 💻

```python
import numpy as np

# Initialize network
w1, b1 = 0.15, 0.40
w2, b2 = 0.45, 0.65
learning_rate = 0.4
epochs = 1000
error_threshold = 0.001

# Training data
x1 = 0.1
T = 0.25  # target

# Sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Forward propagation
def forward_prop(x, w1, b1, w2, b2):
    z1 = w1 * x + b1
    a1 = sigmoid(z1)
    z2 = w2 * a1 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Training loop
for epoch in range(epochs):
    # Forward propagation
    z1, a1, z2, a2 = forward_prop(x1, w1, b1, w2, b2)
    
    # Calculate error
    error = 0.5 * (T - a2)**2
    
    # Stop if error is below threshold
    if error < error_threshold:
        print(f"Converged at epoch {epoch}")
        break
    
    # Backpropagation - calculate gradients
    dE_da2 = -(T - a2)
    da2_dz2 = a2 * (1 - a2)
    
    # Update w2
    dz2_dw2 = a1
    dE_dw2 = dE_da2 * da2_dz2 * dz2_dw2
    w2 = w2 - learning_rate * dE_dw2
    
    # Update b2
    dz2_db2 = 1
    dE_db2 = dE_da2 * da2_dz2 * dz2_db2
    b2 = b2 - learning_rate * dE_db2
    
    # Update w1
    dz2_da1 = w2
    da1_dz1 = a1 * (1 - a1)
    dz1_dw1 = x1
    dE_dw1 = dE_da2 * da2_dz2 * dz2_da1 * da1_dz1 * dz1_dw1
    w1 = w1 - learning_rate * dE_dw1
    
    # Update b1
    dz1_db1 = 1
    dE_db1 = dE_da2 * da2_dz2 * dz2_da1 * da1_dz1 * dz1_db1
    b1 = b1 - learning_rate * dE_db1
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Error: {error}")
        
print("Final parameters:")
print(f"w1: {w1}, b1: {b1}")
print(f"w2: {w2}, b2: {b2}")
```

## Key Takeaways 💡

1. Backpropagation is how neural networks learn by propagating error backwards through the network
2. It uses the chain rule from calculus to calculate how each weight and bias affects the error
3. The process involves:
    - Forward propagation to get predictions
    - Error calculation
    - Backward propagation to update weights and biases
4. Learning rate controls how quickly weights are updated
5. The process repeats until convergence (low error or max iterations)

## Glossary 📖

- **Backpropagation**: Algorithm for calculating gradients in neural networks by propagating error backwards
- **Chain Rule**: Calculus principle for finding derivatives of composite functions
- **Epoch**: One complete pass through the training dataset
- **Forward Propagation**: Process of generating predictions by passing inputs through the network
- **Gradient**: The partial derivatives that indicate how to change weights to reduce error
- **Learning Rate**: Parameter controlling the size of weight updates
- **Sigmoid Function**: Activation function $f(z) = \frac{1}{1+e^{-z}}$ that outputs values between 0 and 1


## Activation Functions

Types of Activation Functions
 - Binary step
 - Linear
 - Sigmoid (remember: can lead to vanishing gradient)
 - Hyperbolic tangent (tanh)
 - Rectified linear unit (ReLU)
 - Leaky ReLU
 - Softmax

