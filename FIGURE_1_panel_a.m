%% This code generates the matrix of FIGURE 1a.

clear;
close all;

% Create the first column (Q = -2) of the matrix. To create the rest 
% columns allow Q to take values from -1.9 to 2 with step 0.1  

Q = -2;             % ring component
G = -2.5:0.1:2.5;   % component of the static weight matrix  
nRepeats = 20;      % number of repetitions


N = 1000;           % number of neurons in the network
T = 10000;          % total simulation time [ms]
dt = 1e-1;          % time step [ms]
nt = round(T/dt);   % number of time steps

k = 2; % dimensionality of the approximant
eta = 2*rand(N, k) - 1; % encoders

mean_rmse = cell(numel(G), 1); % save the mean value of Root Mean Square Error (RMSE) into a cell array

% iterate for all values of G
for gIdx = 1:numel(G)
    tic

    g = G(gIdx);

    % preallocate for this Q value
    target = cell(nRepeats, 1);         % target dynamics of the supervisor
    prediction = cell(nRepeats, 1);     % computed outcome of the supervisor
    mse = cell(nRepeats, 1);            % Mean Square Error between the target and the outcome
    rmse = cell(nRepeats, 1);           % Root Mean Square Error between the target and the outcome
    
    % repeat for "nRepeats" times
    for rep = 1:nRepeats
        tic
        
        fprintf("Running g = %.1f (repeat %d/%d)\n", g, rep, nRepeats);

        %% weight matrix            
        omega = randn(N, N)/sqrt(N);
        omega = omega * g;
        
        % ring couplings
        for j = 1:N-1
            omega(j+1, j) = omega(j+1, j) + Q;
        end
        
        omega(1, N) = omega(1, N) + Q;


        %% supervisor: FitzHugh-Nagumo (FHN)
        mu = 5;     % sets the behavior  
        MD = 1;     % scale system in space 
        TC = 15;    % scale system in time 
        
        % FHN parameters
        a = 0.2;
        b = 0.8;
        tau = 1000;
        Iext = 0.5;
        
        [t, y] = ode45(@(t, y) FHN(a, b, tau, Iext, MD, TC, t, y), 0:dt:TC, [0.1; 0.1]);
        [t, y] = ode45(@(t, y) FHN(a, b, tau, Iext, MD, TC, t, y), 0:dt:T, y(end, :));
        
        kappa = 10;    
        tx = (1:1:nt)*dt; 
        zx(:, 1) = interp1(t, y(:, 1), tx); 
        zx(:, 2) = interp1(t, y(:, 2), tx);
        
        
        store = zeros(nt,10);
        storew = store;
        z = randn(N,1);
        z(1:N/2) = randn(N/2,1);
        z(N/2+1:N) = randn(N/2,1);
            
            
        %% RLS parameters
        alpha = 1; % learning rate for weight change
        P = eye(N)/alpha;  % RLS matrix
        wo = zeros(N,k);
        storez = zeros(nt,k);
        step = 2;
        dn = round(N/10);
        v = 0;
            
            
        %%
        for i = 1:nt 
            r = tanh(z);
            xhat = wo'*r;
            z = z + dt*(-z + omega*r + eta*xhat)/kappa;
            store(i,:) = r(1:dn:N);
            
            err = xhat - zx(i,:)';
            
            if dt*i>1000 & dt*i<T/2
                % RLS learning
                if mod(i, step)==1
                    Pr = P * r;
                    k = Pr / (1 + r' * Pr);
                    P = P - (k * Pr');
                    v = k*err';  
                    wo = wo - v;
                end
            end
            
            storez(i,:) = xhat;
            storew(i,:) = wo(1:dn:N);
        
        end
    

        %% accuracy metrics (after learning)
        target{rep} = zx(nt/2:end, :);
        prediction{rep} = storez(nt/2:end, :);
        
        mse{rep} = mean((target{rep} - prediction{rep}).^2);
        rmse{rep} = sqrt(mse{rep})

        toc
    end

    %% save results 
    fname_prefix = sprintf("g_%0.1f", g);

    writecell(rmse, sprintf("RMSE_Q_%s.txt", fname_prefix));

    %% compute the mean(RMSE)
    rmse_sum = cellfun(@sum, rmse)
    mean_rmse{gIdx} = mean(rmse_sum);

    toc

end

writecell(mean_rmse, "mean_RMSE_Q.txt");










%%
% FitzHugh-Nagumo

function dy = FHN(a, b, tau, Iext, MD, TC, t, y)
    v = y(1);
    w = y(2);
    
    dv = v - (v^3)/3 - w + Iext;
    dw = (v + a - b * w) / tau;

    dy = [dv; dw] * TC * MD;  % scale in time and space
end

