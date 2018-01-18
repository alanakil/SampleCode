
% Call mex AdExNetworkSimDeluxe.c before running this script.

clear
close all

% Number of neurons in each population
Ne=4000;
Ni=1000;
N=Ne+Ni;

% Number of neurons in ffwd layer
Nf=1000;

% Recurrent net connection probabilities
P=[0.05 0.05; 0.05 0.05];

% Ffwd connection probs
Pf=[.05; .05];

% Connection strengths
J=[50 -300; 225 -500]/sqrt(N);
Jf=[180; 135]/sqrt(N);

% Time (in ms) for sim and burn-in for computing means
T=5000;
Tburn=500;

% Time discretization
dt=.1;

% Proportions
qe=Ne/N;
qi=Ni/N;
qf=Nf/N;

% FFwd spike train rate (in kHz)
rf=15/1000;

% No extra stimulus
Istim=[];
JIstim=[];

% Build mean field matrices: O(1)
Q=[qe qi; qe qi];
Qf=[qf; qf];
epsilon=1/sqrt(N);
W=P.*(J*sqrt(N)).*Q;
Wf=Pf.*(Jf*sqrt(N)).*Qf;

% Balanced rate estimate
rBal=-inv(W)*Wf*rf;
disp(sprintf('E and I rates predicted by balance: %.2fHz %.2fHz',1000*rBal(1),1000*rBal(2)))

% Synaptic timescales
tauF=10;
taue=8;
taui=4;

% Number of time bins
Nt=round(T/dt);

% Build Erdos-Renyi connections
tic
[Jinds,Jweights,J0,Jfinds,Jfweights,Jf0,inDegs,outDegs]=GenBlockER(Ne,Ni,Nf,J,Jf,P,Pf);
tGen=toc;
disp(sprintf('\nTime to generate connections: %.2f sec',tGen))

% Make Poisson spike times for ffwd layer
nst=poissrnd(Nf*rf*T);
st=rand(nst,1)*T;
sf=zeros(2,numel(st));
sf(1,:)=sort(st);
sf(2,:)=randi(Nf,1,numel(st)); % neuron indices
clear st;

% AdEx neuron parameters
Cm=1; 
gL=1/15;
EL=-72;
Vth=-50;
Vre=-75;
Vlb=-1000; % Effectively no lower boundary on voltage
tauref=0.5;  
Iapp=0;    % No external input (put E0 term here to use it)
DeltaT=2;  
VT=-55;    
a=0; % EIF
b=0;
tauw=150;

% Need to store params into a vector with E and I params on each row
NeuronParams=[Cm gL EL Vth Vre Vlb tauref Iapp DeltaT VT a b tauw;
    Cm gL EL Vth Vre Vlb tauref Iapp DeltaT VT a b tauw];

% Vector of neuron types: 0 for exc, 1 for inh.
NeuronType=[zeros(1,Ne) ones(1,Ni)];

% Compute PSP amplitudes
v=GetPSP(Cm,taue,0,J(1,1),gL,EL,100,dt);EEPSP=max(v)-min(v);
v=GetPSP(Cm,taue,0,J(2,1),gL,EL,100,dt);IEPSP=max(v)-min(v);
v=GetPSP(Cm,taui,0,J(1,2),gL,EL,100,dt);EIPSP=max(v)-min(v);
v=GetPSP(Cm,taui,0,J(2,2),gL,EL,100,dt);IIPSP=max(v)-min(v);
v=GetPSP(Cm,tauF,0,Jf(1),gL,EL,100,dt);EFPSP=max(v)-min(v);
v=GetPSP(Cm,tauF,0,Jf(2),gL,EL,100,dt);IFPSP=max(v)-min(v);
PSPs=[EEPSP EIPSP; IEPSP IIPSP];
PSPFs=[EFPSP; IFPSP];

% Random initial voltages
V0=rand(N,1)*(VT-Vre)+Vre;

% Maximum number of spikes for all neurons
% in simulation. Make it 50Hz across all neurons
maxns=ceil(.05*N*T);

% Indices of neurons to record currents, voltages
nrecord=10;
Irecord=sort([randperm(Ne,nrecord) randperm(Ni,nrecord)+Ne]);
nskiprecord=5;

% Run sim
tic;
[s,alphaf,alphae,alphai,~,~,~,malphaf,malphae,malphai,~,~]=AdExNetworkSimDeluxe(N,Nf,NeuronType,T,sf,Istim,JIstim,Jinds,Jweights, ...
                                                            J0,Jfinds,Jfweights,Jf0,NeuronParams,tauF,taue,taui, ...
                                                            V0,dt,maxns,Tburn,[],Irecord,nskiprecord);


% Get rid of unused parts of s (which have 0 for spike times)
s=s(:,s(1,:)>0);
tSim=toc;
disp(sprintf('\nTime for simulation: %.2f sec',tSim))

% Make a raster plot
figure
plot(s(1,s(1,:)<2000 & s(2,:)<500),s(2,s(1,:)<2000 & s(2,:)<500),'.')
xlabel('time (ms)')
ylabel('Neuron index')

% Mean rate of each neuron
reSim=hist(s(2,s(1,:)>Tburn & s(2,:)<=Ne),1:Ne)/(T-Tburn);
riSim=hist(s(2,s(1,:)>Tburn & s(2,:)>Ne)-Ne,1:Ni)/(T-Tburn);

% Mean rate over E and I pops
rSim=[mean(reSim) mean(riSim)];
disp(sprintf('\nE and I rates from sims: %.2fHz %.2fHz',1000*rSim(1),1000*rSim(2)))


% Get mean CV over a bunch of neurons
% If this is very small, network is oscillatory
% If it's >1, you might have rate chaos.
nCV=200;
CVs=zeros(nCV,1);
for j=1:nCV
    % Only compute CV if rate is >.5Hz
    if(reSim(j)>.5/1000)
       ISIs=diff(s(1,s(2,:)==j));
       CVs(j)=std(ISIs)/mean(ISIs);
    end
end
mCV=mean(CVs);

disp(sprintf('\nMean CV: %.2f',mCV))


