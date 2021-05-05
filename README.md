# dlmstan

This package demonstrates how to use Stan to fit dynamic linear models of form

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;x_{k&plus;1}&space;&&space;=&space;A(\theta)x_k&space;&plus;&space;B(\theta)u_k&space;&&space;&plus;&space;N(0,&space;Q(\theta))\\&space;y_{k&plus;1}&space;&&space;=&space;C(\theta)x_{k&plus;1}&space;&&space;&plus;&space;N(0,&space;R(\theta))&space;\end{}" title="\begin{align*} x_{k+1} & = A(\theta)x_k + B(\theta)u_k & + N(0, Q(\theta))\\ y_{k+1} & = C(\theta)x_{k+1} & + N(0, R(\theta)) \end{}" />

That is, we fit some static parameters of a linear state space model, including possibly parameters for the model and observation noise. In addition, we show how to sample the states given the parameters efficiently, inspired by the blog post in <url>http://www.juhokokkala.fi/blog/posts/kalman-filter-style-recursion-to-marginalize-state-variables-to-speed-up-stan-inference/</url> (here we consider a more general case than in the blog post).

Note that there is also a function already available in Stan for DLM fitting, see <url>https://mc-stan.org/docs/2_26/functions-reference/gaussian-dynamic-linear-models.html</url>. The difference here is that we include the "forcing term" B(&theta;)u<sub>k</sub> and consider also the random sampling of the states given the parameters.

### Theory

For a linear state space system, the likelihood of the parameters can be efficiently calculated by integrating out the state variables using a Kalman Filter recursion. Let's first look at the general case: 

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;x_k&space;&&space;\sim&space;p(x_k&space;|&space;x_{k-1},&space;\theta)&space;\\&space;y_k&space;&&space;\sim&space;p(y_k&space;|&space;x_k)&space;\end{align*}" title="\begin{align*} x_k & \sim p(x_k | x_{k-1}, \theta) \\ y_k & \sim p(y_k | x_k) \end{align*}" />

The likelihood of the parameters can be calculated using the chain rule of joint probability:

<img src="https://latex.codecogs.com/gif.latex?p(y_{1:n}&space;|&space;\theta)&space;=&space;p(y_n&space;|&space;y_{1:n-1},&space;\theta)p(y_{1:n-1}&space;|&space;y_{1:n-2},&space;\theta)&space;\cdots&space;p(y_2&space;|&space;y_1,&space;\theta)&space;p(y_1&space;|&space;\theta)" title="p(y_{1:n} | \theta) = p(y_n | y_{1:n-1}, \theta)p(y_{1:n-1} | y_{1:n-2}, \theta) \cdots p(y_2 | y_1, \theta) p(y_1 | \theta)" />
