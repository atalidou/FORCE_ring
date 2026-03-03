%% This code generates FIGURE 4 sin 5Hz.

clear;
close all;

rng(1)

Q = -2;     % ring component
G = 0;      % component of the static weight matrix

%% simulation parameters
dt = 0.00001;       % time step [s]
T = 30;             % total simulation time [s]
nt = round(T/dt);   % number of time steps

N = 1000;           % number of neurons in the network
        
%% LIF parameters
tref = 0.03;        % refractory time constant [s]
tm = 0.01;          % membrane time constant [s]
td = 0.03;          % decay time constant [s]   
tr = 0.002;         % synaptic rise time constant [s]

vreset = -65;       % voltage reset value [mV]
vpeak = -40;        % voltage peak value [mV]
         
%% RLS parameters
alpha = dt*0.1;             % learning rate for weight change
Pinv = eye(N)*alpha;        % RLS correlation weight matrix initialization
imin = round(5/dt);         % time before starting the RLS
icrit = round(15/dt);       % end simulation at this time step
step = 10;                  % optimize with RLS every "step" steps
        
%% supervisor: sin 5Hz
zx = sin(2*pi*(1:nt)*dt*5)';

%% weight matrix
OMEGA = randn(N,N)/sqrt(N) * G;

% ring couplings
for j = 1:N
    % index of 1st neighbor
    idx1 = mod(j, N) + 1;
    
    OMEGA(idx1, j) = OMEGA(idx1, j) + Q;
end        
        
%% storage variables
IPSC = zeros(N, 1);          % post-synaptic current
r = zeros(N, 1);             % filtered rate variable
h = zeros(N,1);              % storage variable for filtered firing rates
hr = zeros(N, 1);            % second filtered rate variable
tspike = zeros(4*nt, 2);     % spike times: [neuron index, time]
k = min(size(zx));           % dimensionality of the approximant
z = zeros(k, 1);             % approximant (output)

m = 10;
REC_decoders = zeros(nt, m);         % store subset of decoders
REC_voltage = zeros(nt, m);          % store subset of voltage
REC_filtered_rates = zeros(nt, m);   % store subset of filtered rates

BPhi = zeros(N, k);          % initial decoder (FORCE learning)
current = zeros(nt, k);      % store the output (approximant)

J = 60;
E = (2*rand(N, k)-1);        % encoders
E = E*J;                     % part of the weight matrix
BIAS = vpeak + 1;            % bias current
ns = 0;                      % number of spikes

v = vreset + rand(N, 1)*(30 - vreset);       % initialize membrane voltages

tlast = zeros(N, 1);         % last spike time for refractory period control
        
tic
%% start simulation
for i = 1:nt

    % compute neuronal current: synaptic + external input + bias
    I = IPSC + E*z + BIAS;

    % update voltage: account for refractory period
    dv = (dt*i > tlast + tref).*(-v + I)/tm;
    v = v + dt*dv;

    % find neurons that reached or exceeded the spike threshold
    index = find(v >= vpeak);

    %% record spike times and schedule delayed contributions
    if ~isempty(index)
        JD = sum(OMEGA(:,index),2);     % compute the increase in current due to spiking  
        tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i];
        ns = ns + length(index);        % total number of spikes so far
    end

    %% update refractory times for neurons that spiked now
    tlast(v >= vpeak) = dt*i;

    %% synapse for double exponential
    IPSC = IPSC*exp(-dt/tr) + h*dt;
    h = h*exp(-dt/td) + JD*(length(index)>0)/(tr*td);  % integrate the current
    
    r = r*exp(-dt/tr) + hr*dt; 
    hr = hr*exp(-dt/td) + (v>=vpeak)/(tr*td);
   

    %% FORCE learning
    z = BPhi' * r;    % network output
    err = z - zx(i);  % error

    % RLS
    if mod(i, step) == 1
        if i > imin
            if i < icrit
                cd = Pinv*r;
                BPhi = BPhi - (cd * err');
                Pinv = Pinv -(cd * cd')/( 1 + r' * cd);
            end
        end
    end

    %% bring spiking neurons back to reset value
    v = v + (30 - v).*(v >= vpeak);
    REC_voltage(i, :) = v(1:m);            % record random voltage
    v = v + (vreset - v) .* (v >= vpeak);  % reset voltage

    current(i, :) = z;
    REC_decoders(i, :) = BPhi(1:m);
    REC_filtered_rates(i, :) = r(1:m);

    %% plots
    if mod(i, round(0.5/dt)) == 1

        drawnow;

        % raster plot
        figure(1)
        plot(tspike(1:ns, 2), tspike(1:ns, 1), 'k.');
        xlim([dt*i-5, dt*i])
        ylim([0, 200])

        % target signal and approximant  
        figure(2)
        drawnow 
        plot(dt*(1:i), zx(1:i), 'k--', 'LineWidth', 2)
        hold on
        plot(dt*(1:i), current(1:i), 'LineWidth', 2)
        hold off
        xlim([dt*i-5, dt*i])
        
        % decoders
        figure(5)
        plot(dt*(1:i), REC_decoders(1:i, 1:10), '.')

    end


end % end for loop

        

