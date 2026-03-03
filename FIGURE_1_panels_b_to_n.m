%% This code generates FIGURE 1b-n.

clear;
close all;

%% Point #1
Q = 1.3;    % ring component
G = 0;      % component of the static weight matrix

% %% Point #2
% Q = -0.7;    % ring component
% G = 1;      % component of the static weight matrix

% %% Point #3
% Q = 0;    % ring component
% G = -1;      % component of the static weight matrix

% %% Point #4
% Q = 1.8;    % ring component
% G = -1;      % component of the static weight matrix


rng(1);

N = 1000;           % number of neurons in the network
T = 10000;          % total simulation time [ms]
dt = 1e-1;          % time step [ms]
nt = round(T/dt);   % number of time steps

k = 2;                      % dimensionality of the approximant
eta = 2*rand(N, k) - 1;     % encoders
    

%% weight matrix
omega = randn(N, N)/sqrt(N);
omega = omega*G;

% ring couplings
for j = 1:N-1
    omega(j+1,j) = omega(j+1,j) + Q;
end

omega(1,N) = omega(1,N) + Q;

        
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


store = zeros(nt, 100); 
storew = store;
z = randn(N,1);
z(1:N/2) = randn(N/2,1);
z(N/2+1:N) = randn(N/2,1);
            
            
%% RLS parameters
alpha = 1; % learning rate for weight change
P = eye(N)/alpha;  % RLS matrix
wo = zeros(N,k);
storez = zeros(nt,k);

store_r = zeros(nt, N);   % neuron activities r(t)

step = 2;
dn = round(N/100); 
v = 0;
            
            
%%
for i = 1:nt 
    r = tanh(z);
    xhat = wo'*r;
    z = z + dt*(-z + omega*r + eta*xhat)/kappa;
    store(i,:) = r(1:dn:N);
    % store activities for all neurons
    store_r(i,:) = r';

    err = xhat - zx(i,:)';
    
    if dt*i>1000 & dt*i<T/2
        % RLS learning
        if mod(i,step)==1
            Pr = P * r;
            k = Pr / (1 + r' * Pr);
            P = P - (k * Pr');
            v = k*err';  
            wo = wo - v;
        end
    end
    
    storez(i,:) = xhat;
    storew(i,:) = wo(1:dn:N);


    %% plots
    if mod(i,1000)==1
        
        figure(1)
        subplot(3,1,1)
    
        for j = 1:10
            plot((1:i)*dt,store(1:i,j)+2*j), hold on 
        end

        hold off
        subplot(3,1,2)
        plot((1:i)*dt,zx(1:i,:)), hold on 
        plot((1:i)*dt,storez(1:i,:)), hold off
        
        subplot(3,1,3)
        plot((1:i)*dt,storew(1:i,:))
        drawnow
        figure(4)
        plot(store(1:i,1),store(1:i,3))
        drawnow
        
    end

end


%%

% pre-learning indices
pre_idx = find(tx < 1000);

% Extract only pre-learning activity for first 200 neurons
store_pre = store_r(pre_idx, 1:200);

% Time window for plotting
t_pre = tx(pre_idx);

% Plot
figure(31);
imagesc(t_pre, 1:200, store_pre');   % transpose: neurons on y-axis
xlabel('Time');
ylabel('Neuron index (1to200)');
title('Firing rates of the first 200 neurons (pre-learning only)');
colorbar;
set(gca, 'YDir', 'normal');
colormap('parula');  


%%
post_idx = find(tx > T/2 + 1 & tx < T/2 + 1001);

% Extract only post-learning activity for first 200 neurons
store_post = store_r(post_idx, 1:200);

% Time window for plotting
t_post = tx(post_idx);

% Plot
figure(41);
imagesc(t_post, 1:200, store_post');   % transpose: neurons on y-axis
xlabel('Time');
ylabel('Neuron index (1to200)');
title('Firing rates of the first 200 neurons (post-learning only)');
colorbar;
set(gca, 'YDir', 'normal');
colormap('parula');  



%% eigenvalues
figure(40)
Z = eig(omega + eta*wo');   % eigenvalues after learning 
Z2 = eig(omega);            % eigenvalues before learning 
plot(Z2,'r.'), hold on 
plot(Z,'k.') 
legend('Pre-Learning','Post-Learning')
xlabel('Re \lambda')
ylabel('Im \lambda')








%%
% FitzHugh-Nagumo

function dy = FHN(a, b, tau, Iext, MD, TC, t, y)
    v = y(1);
    w = y(2);
    
    dv = v - (v^3)/3 - w + Iext;
    dw = (v + a - b * w) / tau;

    dy = [dv; dw] * TC * MD;  % scale in time and space
end

