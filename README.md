# dlmstan - Dynamic Linear Models fitted with Stan

This package demonstrates how to use Stan to fit dynamic linear models of form

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;x_{k&plus;1}&space;&&space;=&space;A(\theta)x_k&space;&plus;&space;B(\theta)u_k&space;&&space;&plus;&space;N(0,&space;Q(\theta))\\&space;y_{k&plus;1}&space;&&space;=&space;K(\theta)x_{k&plus;1}&space;&&space;&plus;&space;N(0,&space;R(\theta))&space;\end{}" title="\begin{align*} x_{k+1} & = A(\theta)x_k + B(\theta)u_k & + N(0, Q(\theta))\\ y_{k+1} & = K(\theta)x_{k+1} & + N(0, R(\theta)) \end{}" />

That is, we fit some static parameters of a linear state space model, including possibly parameters for the model and observation noise. In addition, we show how to sample the states given the parameters efficiently, inspired by the blog post in <url>http://www.juhokokkala.fi/blog/posts/kalman-filter-style-recursion-to-marginalize-state-variables-to-speed-up-stan-inference/</url> (here we consider a more general case than in the blog post).

Note that there is also a function already available in Stan for DLM fitting, see <url>https://mc-stan.org/docs/2_26/functions-reference/gaussian-dynamic-linear-models.html</url>. The difference here is that we include the "forcing term" B(&theta;)u<sub>k</sub> and consider also the random sampling of the states given the parameters.

The code runs with `CmdStanPy`, which is a thin python-wrapper around `CmdStan`, the command line interface to Stan. In addition to `CmdStanPy`, you'll need some standard python stuff like `numpy`, `pandas` and `matplotlib`.

### Theory

For a linear state space system, the likelihood of the parameters can be efficiently calculated by integrating out the state variables using a Kalman Filter recursion. The likelihood of the parameters can be calculated using the chain rule of joint probability:

<img src="https://latex.codecogs.com/gif.latex?p(y_{1:n}&space;|&space;\theta)&space;=&space;p(y_n&space;|&space;y_{1:n-1},&space;\theta)p(y_{1:n-1}&space;|&space;y_{1:n-2},&space;\theta)&space;\cdots&space;p(y_2&space;|&space;y_1,&space;\theta)&space;p(y_1&space;|&space;\theta)" title="p(y_{1:n} | \theta) = p(y_n | y_{1:n-1}, \theta)p(y_{1:n-1} | y_{1:n-2}, \theta) \cdots p(y_2 | y_1, \theta) p(y_1 | \theta)" />

The individual predictive distributions can be calculated in the linear case recursively as follows (dropping the parameter dependency from the matrices here for simplicity), starting from <img src="https://latex.codecogs.com/gif.latex?x_k&space;|&space;y_{1:k},&space;\theta&space;\sim&space;N(x_k^{est},&space;C_k^{est})" title="x_k | y_{1:k}, \theta \sim N(x_k^{est}, C_k^{est})" />:

1) Predict the state forward: <img src="https://latex.codecogs.com/gif.latex?x_k&space;|&space;y_{1:k-1},&space;\theta&space;\sim&space;N(x_k^p,&space;C_k^p)" title="x_k | y_{1:k-1}, \theta \sim N(x_k^p, C_k^p)" /> where the predicted mean and covariance are 

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;x_k^p&space;&&space;=&space;A&space;x_{k-1}^{est}&space;&plus;&space;B&space;u_k&space;\\&space;C_k^p&space;&&space;=&space;A&space;C_k^{est}&space;A^T&space;&plus;&space;Q&space;\end{}" title="\begin{align*} x_k^p & = A x_{k-1}^{est} + B u_k \\ C_k^p & = A C_k^{est} A^T + Q \end{}" />

2) Update the state with the new observation: <img src="https://latex.codecogs.com/gif.latex?x_k&space;|&space;y_{1:k},&space;\theta&space;\sim&space;N(x_k^{est},&space;C_k^{est})" title="x_k | y_{1:k}, \theta \sim N(x_k^{est}, C_k^{est})" /> where the posterior mean and covariance are

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;x_k^{est}&space;&&space;=&space;x_k^p&space;&plus;&space;G_k(y_k-Kx_k^p)&space;\\&space;C_k^{est}&space;&&space;=&space;C_k^p&space;-&space;G_k&space;K&space;C_k^p&space;\\&space;G_k&space;&&space;=&space;C_k^p&space;K^T&space;(KC_k^pK^T&space;&plus;&space;R)^{-1}&space;\end{}" title="\begin{align*} x_k^{est} & = x_k^p + G_k(y_k-Kx_k^p) \\ C_k^{est} & = C_k^p - G_k K C_k^p \\ G_k & = C_k^p K^T (KC_k^pK^T + R)^{-1} \end{}" />

3) Calculate the posterior predictive distribution: <img src="https://latex.codecogs.com/gif.latex?y_k&space;|&space;y_{1:k-1},&space;\theta&space;\sim&space;N(Kx_k^p,&space;K&space;C_k^p&space;K^T&space;&plus;&space;R)" title="y_k | y_{1:k-1}, \theta \sim N(Kx_k^p, K C_k^p K^T + R)" />

Now the final likelihood is the combination of the above densities, and we can run MCMC to sample the posterior for &theta;. Along the posterior sampling, we can get also samples of the states x<sub>1:T</sub> using the following backward recursion:

1) sample x<sub>T</sub> from <img src="https://latex.codecogs.com/gif.latex?p(x_T&space;|&space;y_{1:T},&space;\theta)" title="p(x_T | y_{1:T}, \theta)" />, which is just the filtering distribution for the last state, <img src="https://latex.codecogs.com/gif.latex?N(x_T^{est},&space;C_T^{est})" title="N(x_T^{est}, C_T^{est})" />
2) sample from <img src="https://latex.codecogs.com/gif.latex?p(x_{T-1}&space;|&space;x_T,&space;y_{1:T},&space;\theta)" title="p(x_{T-1} | x_T, y_{1:T}, \theta)" />
3) sample from <img src="https://latex.codecogs.com/gif.latex?p(x_{T-2}&space;|&space;x_{T-1},&space;x_T,&space;y_{1:T},&space;\theta)" title="p(x_{T-2} | x_{T-1}, x_T, y_{1:T}, \theta)" />
4) ....

That is, we end up having to sample from densities like 

<img src="https://latex.codecogs.com/gif.latex?p(x_k&space;|&space;x_{k&plus;1:T},&space;y_{1:T},&space;\theta)&space;=&space;p(x_k&space;|&space;x_{k&plus;1},&space;y_{1:k},&space;\theta)&space;\propto&space;p(x_{k&plus;1}&space;|&space;x_k)&space;p(x_k&space;|&space;y_{1:k},&space;\theta)" title="p(x_k | x_{k+1:T}, y_{1:T}, \theta) = p(x_k | x_{k+1}, y_{1:k}, \theta) \propto p(x_{k+1} | x_k) p(x_k | y_{1:k}, \theta)" />

where the first equation follows from the Markovian model (conditional independence) and the second equation is the Bayes formula. The components in the final expressions are Gaussians:

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;x_{k&plus;1}&space;&&space;\sim&space;N(Ax_k&plus;Bu_k,&space;Q)&space;\\&space;x_k&space;|&space;y_{1:k},&space;\theta&space;&&space;\sim&space;N(x_k^{est},&space;C_k^{est})&space;\end{}" title="\begin{align*} x_{k+1} & \sim N(Ax_k+Bu_k, Q) \\ x_k | y_{1:k}, \theta & \sim N(x_k^{est}, C_k^{est}) \end{}" />

We can think of the first one as "likelihood" with x<sub>k+1</sub> as "data" and the latter as prior, which gives us the posterior <img src="https://latex.codecogs.com/gif.latex?x_k&space;|&space;x_{k&plus;1},&space;y_{1:k},&space;\theta&space;\sim&space;N(\mu_k,&space;\Sigma_k)" title="x_k | x_{k+1}, y_{1:k}, \theta \sim N(\mu_k, \Sigma_k)" />, where

<img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\Sigma_k&space;&&space;=&space;(A^T&space;Q^{-1}&space;A&space;&plus;&space;(C_k^{est})^{-1})^{-1}&space;\\&space;\mu_k&space;&&space;=&space;\Sigma_k&space;(A^T&space;Q^{-1}&space;(x_{k&plus;1}-Bu_k)&plus;(C_k^{est})^{-1}x_k^{est})&space;\end{}" title="\begin{align*} \Sigma_k & = (A^T Q^{-1} A + (C_k^{est})^{-1})^{-1} \\ \mu_k & = \Sigma_k (A^T Q^{-1} (x_{k+1}-Bu_k)+(C_k^{est})^{-1}x_k^{est}) \end{}" />

That is, if we store all the results from the Kalman filter forward pass, we can calculate a random sample given &theta; using the pre-calculated results.

### Stan code

The general Stan code for sampling the parameters and states given the parameters using the above trick is given in <url>https://github.com/solbes/dlmstan/blob/main/dlm.stan</url>. The code does not run on it's own, the user needs to write the functions `build_A`, `build_B` etc for the application in question, which can be done using the `functions` block in Stan, see the examples below. The user also needs to feed in data in a standard format, see again the examples.

Some comments about the code:
- User writes the `functions` block for the problem at hand, which is then appended by the general code `dlm.stan`, see the examples.
- If matrix `B` is not present in the application, the user needs to still feed in a dummy zero matrix, see the first example
- The code separates two types of parameters, normal "model parameters" and noise parameters that enter the covariance matrices `Q` and `R`. Normal priors are given for both, whose parameters can be given in the data, see the examples.
- Initial values for the states need to be given in the data, see the examples.
- In the DLM code, the matrix inversion lemma is used to avoid explicitly inverting the filtering covariance matrices.

### Examples

1) Nile river, estimating 3 noise parameters: <url>https://github.com/solbes/dlmstan/blob/main/examples/stan_dlm_nile.ipynb</url>
2) Beer cooling, 2 model parameters and 2 noise parameters: <url>https://github.com/solbes/dlmstan/blob/main/examples/stan_dlm_beer.ipynb</url>
