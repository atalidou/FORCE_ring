%% This code generates FIGURE 2.

clear;
close all;

Q = 1.5;    % ring component
G = 0;      % component of the static weight matrix

rng(1)

N = 2000;           % number of neurons in the network
T = 17000;          % total simulation time [ms]
dt = 1e-1;          % time step [ms]
nt = round(T/dt);   % number of time steps

k = 3;                      % dimensionality of the approximant
eta = 2*rand(N, k) - 1;     % encoders

%% weight matrix
omega = randn(N, N)/sqrt(N);
omega = omega*G;

% ring couplings
for j = 1:N-1
    omega(j+1,j) = omega(j+1,j)+Q;
end

omega(1,N) = omega(1,N) + Q;

%% supervisor: Lorenz attractor
sigma = 10;
rho = 28;
beta = 8/3;
tau = 0.02;

[t,y] = ode45(@(t,y) tau*lorenz_sys(y,sigma,rho,beta),0:dt:T,[-1,0,1]');

sup(:,1) = (y(:,3)-mean(y(:,3)))/std(y(:,3));
sup(:,2) = (y(:,1)-mean(y(:,1)))/std(y(:,1));
sup(:,3) = (y(:,2)-mean(y(:,2)))/std(y(:,2));

kappa = 5;

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
    
    err = xhat - sup(i,:)';
    
    if dt*i>1000 & dt*i<T/2
        % RLS learning
        if mod(i,step)==1
            Pr = P * r;
            k = Pr / (1 + r' * Pr);
            P = P - (k * Pr');
            %grad = r*epsilon*err';
            v = k*err';  
            %v = beta * v + (1 - beta) * grad;
            wo = wo - v;
        end
    end
    
    storez(i,:) = xhat;
    storew(i,:) = wo(1:dn:N);


    %% plot
    if mod(i,1000)==1
        
        figure(1)
        subplot(3,1,1)
    
        for j = 1:10
            plot((1:i)*dt,store(1:i,j)+2*j), hold on 
        end

        hold off
        subplot(3,1,2)
        plot((1:i)*dt,sup(1:i,:)), hold on 
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

[pks,loc] = findpeaks(storez(nt/2:nt,1),'minpeakdistance',20,'minpeakheight',0);
[pks2,loc2] = findpeaks(sup(nt/2:nt,1),'minpeakdistance',20,'minpeakheight',0);

figure(2)
subplot(1,2,1)
plot(storez(nt/2:nt,1)), hold on
plot(loc,pks,'k.'), hold off
subplot(1,2,2)
plot(pks(1:end-1),pks(2:end),'k.'), hold on 
plot(pks2(1:end-1),pks2(2:end),'r.'), hold off


%% eigenvalues before and after learning
figure(40)
Z = eig(OMEGA+E*BPhi');     % eigenvalues after learning 
Z2 = eig(OMEGA);            % eigenvalues before learning 
plot(Z2,'r.'), hold on 
plot(Z,'k.') 
legend('Pre-Learning','Post-Learning')
xlabel('Re \lambda')
ylabel('Im \lambda')










%% 
% Lorenz function

function dX = lorenz_sys(X,sigma,rho,beta)
    x = X(1,1);
    y = X(2,1);
    z = X(3,1); 

    dX(1,1) = sigma*(y - x);
    dX(2,1) = x*(rho - z) - y;
    dX(3,1) = x*y- beta*z;

end


