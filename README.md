# dlmstan

This package demonstrates how to use Stan to fit dynamic linear models of form

x<sub>k+1</sub> = A(&theta;)x<sub>k</sub> + B(&theta;)u<sub>k</sub> + N(0,Q(&theta;))

y<sub>k+1</sub> = C(&theta;)x<sub>k+1</sub> + N(0,R(&theta;)).

That is, we fit some static parameters of a linear state space model, including possibly parameters for the model and observation noise. In addition, we show how to sample the states given the parameters efficiently, inspired by the blog post in <url>http://www.juhokokkala.fi/blog/posts/kalman-filter-style-recursion-to-marginalize-state-variables-to-speed-up-stan-inference/</url> (here we consider a more general case than in the blog post).

Note that there is also a function already available in Stan for DLM fitting, see <url>https://mc-stan.org/docs/2_26/functions-reference/gaussian-dynamic-linear-models.html</url>. The difference here is that we include the "forcing term" B(&theta;)u<sub>k</sub> and consider also the random sampling of the states given the parameters.
