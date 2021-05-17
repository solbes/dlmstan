data {
    // dimensions
    int N_obs;
    int N_theta;
    int N_noise_theta;
    int state_dim;
    int input_dim;
    int obs_dim;
    
    // observations
    vector[obs_dim] Y_obs[N_obs];
    vector[input_dim] U_obs[N_obs];
    
    // initial mean and covariance
    matrix[state_dim, state_dim] P0;
    vector[state_dim] m0;
    
    // normal prior for parameters
    matrix[N_theta, N_theta] theta_Sig;
    vector[N_theta] theta_mu;
    
    // normal prior for noise parameters
    matrix[N_noise_theta, N_noise_theta] noise_Sig;
    vector[N_noise_theta] noise_mu;
}
parameters {
    vector[N_theta] theta;
    vector<lower=0>[N_noise_theta] noise_theta;
}
transformed parameters {

    vector[obs_dim] y_pred[N_obs];
    matrix[obs_dim, obs_dim] S_pred[N_obs];
    vector[state_dim] m[N_obs];
    matrix[state_dim, state_dim] P[N_obs];
    matrix[state_dim, state_dim] A;
    matrix[state_dim, input_dim] B;
    matrix[obs_dim, state_dim] C;
    matrix[obs_dim, obs_dim] R;
    matrix[state_dim, state_dim] Q;
    
    vector[state_dim] m_i;
    matrix[state_dim, state_dim] P_i;
    vector[state_dim] m_pred;
    matrix[state_dim, state_dim] P_pred;
    matrix[state_dim, obs_dim] G;
    
    // build the matrices
    A = build_A(theta);
    B = build_B(theta);
    C = build_C(theta);
    Q = build_Q(noise_theta);
    R = build_R(noise_theta);
    
    m_i = m0;
    P_i = P0;
    
    for (i in 1:N_obs) {
    
        // predicted means and variances for state
        m_pred = A*m_i + B*U_obs[i];
        P_pred = A*P_i*A' + Q;
        
        // save predicted mean and var for obs
        y_pred[i] = C*m_pred;
        S_pred[i] = C*P_pred*C' + R;
    
        // update with KF
        G = (P_pred*C')/S_pred[i];
        m_i = m_pred+G*(Y_obs[i]-y_pred[i]);
        P_i = P_pred-G*C*P_pred;
        
        // store for later use (state sampling)
        m[i] = m_i;
        P[i] = P_i;
    }
}
model {
    // prior (weakly informative)
    theta ~ multi_normal(theta_mu, theta_Sig);
    noise_theta ~ multi_normal(noise_mu, noise_Sig);

    // likelihood
    for (i in 1:N_obs) {
        Y_obs[i] ~ multi_normal(y_pred[i], S_pred[i]);
    }   
}
generated quantities {
    vector[state_dim] x_samples[N_obs];
    matrix[state_dim, state_dim] AP_k;
    matrix[state_dim, state_dim] AtQinv = A'/Q;
    int k;
    vector[state_dim] mu_k;
    matrix[state_dim, state_dim] Sig_k;
    
    //Sampling x
    x_samples[N_obs] = multi_normal_rng(m[N_obs], P[N_obs]);
    for (i in 1:N_obs-1) {
        k = N_obs-i;
        AP_k = A*P[k];
        Sig_k = P[k]-AP_k'*((AP_k*A'+Q)\AP_k);
        mu_k = Sig_k*(AtQinv*(x_samples[k+1]-B*U_obs[k]) + P[k]\m[k]);
        x_samples[k] = multi_normal_rng(mu_k,Sig_k);
    }
}
