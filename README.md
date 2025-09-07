# This is a note focusing on machine learning enigeering 
This project covers a lot of machine learning knowledge and project content in the learning process, learning by doing. Any comments and discussions are welcome!
### So let's start with the simplest neural network below!

## Steps in Writing a Neural Network

When building a neural network from scratch, the process can be divided into several key steps:

1. **Define Activation Functions**  
   Implement common activation functions (e.g., Sigmoid, ReLU, Tanh) that introduce non-linearity into the model.
## Activation Functions

## Activation Functions

Activation functions introduce non-linearity into neural networks and help determine whether a neuron should be activated.  
They map input signals into specific ranges (e.g., probabilities or bounded values), making deep learning models expressive and powerful.

---

## Activation Functions

Activation functions introduce non-linearity into neural networks and help determine whether a neuron should be activated.  
They map input signals into specific ranges (e.g., probabilities or bounded values), making deep learning models expressive and powerful.

---

### 1. Sigmoid
**Formula:**  
\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

- Output range: (0, 1)  
- Commonly used in **binary classification**.  
- Limitation: suffers from the **vanishing gradient problem**.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg" width="400"/>
</p>

---

### 2. Tanh
**Formula:**  
\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]

- Output range: (-1, 1)  
- Zero-centered, better than sigmoid for convergence.  
- Still prone to **vanishing gradients**.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/cb/Activation_tanh.svg" width="400"/>
</p>

---

### 3. ReLU (Rectified Linear Unit)
**Formula:**  
\[
f(x) = \max(0, x)
\]

- Keeps positive values unchanged, replaces negatives with 0.  
- Helps **mitigate vanishing gradients** and speeds up training.  
- Limitation: may lead to **dead neurons** (outputs stuck at 0).

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/6c/Rectifier_and_softplus_functions.svg" width="400"/>
</p>

---

### 4. Leaky ReLU
**Formula:**  
\[
f(x) = 
\begin{cases} 
x & \text{if } x \geq 0 \\ 
\alpha x & \text{if } x < 0 
\end{cases}
\]

- Introduces a small slope (\(\alpha \approx 0.01\)) for negative values.  
- Prevents the **dead neuron problem** in ReLU.  

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/ae/Activation_prelu.svg" width="400"/>
</p>

---

### 5. ELU (Exponential Linear Unit)
**Formula:**  
\[
f(x) = 
\begin{cases} 
x & \text{if } x \geq 0 \\ 
\alpha (e^x - 1) & \text{if } x < 0 
\end{cases}
\]

- Smooths the negative part compared to ReLU.  
- Helps reduce bias shift and improve training.



---

### 6. Softmax
**Formula:**  
\[
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
\]

- Converts logits into a **probability distribution**.  
- Used in the **output layer for multi-class classification**.





   

3. **Implement Loss Functions**  
   Write loss functions (e.g., Mean Squared Error, Cross-Entropy) to measure the difference between predicted and actual outputs.

4. **Design Layers**  
   Construct layers such as input, hidden, and output layers, and define how neurons are connected.

5. **Build Optimizers**  
   Implement optimization algorithms (e.g., Gradient Descent, Adam) to update the weights and biases efficiently.

6. **Combine and Train the Network**  
   Integrate activation functions, loss functions, layers, and optimizers into a complete neural network.  
   Train the model by iteratively forward-propagating inputs, calculating the loss, backpropagating gradients, and updating parameters.
