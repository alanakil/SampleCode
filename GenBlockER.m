
% FUNCTION: BuildJsBlockER

% INPUT
%     Ne = Number of excitatory neurons in recurrent network
%     Ni = Number of inhibitory neurons in recurrent network
%     Nf = Number of feedforward neurons projecting to recurrent network
%     J = 2x2 matrix of mean-field recurrent post-syn potentials (PSP's)
%     Jf = 2x1 matrix of mean-field feedforward PSP's
%     P = 2x2 matrix of mean-field recurrent connection probabilities
%     Pf = 2x1 matrix of mean-field feedforward connection probabilities

% OUTPUT
%     Jinds = unweighted (binary) recurrent Adjacency List
%     Jweights = weighted recurrent Adjacency List
%     J0 = recurrent Adjacency indexing list
%     
%     Jfinds = unweighted (binary) feedforward Adjacency List
%     Jfweights = weighted feedforward Adjacency List
%     Jf0 = feedforward Adjacency indexing list
%     
%     Syntax to extract list of post-synaptic adjacencies (connections) for 
%     neuron i is
%         Jinds(J0(i):( J0(i+1)-1 ))
%     i.e., the non-zero elements of column i in the adjacency matrix
%     i.e., the list of neurons that neuron i connects to

% Note: Degrees are measurable by diff(J0)

function [Jinds,Jweights,J0,Jfinds,Jfweights,Jf0,inDegList,outDegList] = GenBlockER(Ne,Ni,Nf,...
    J,Jf,p,pf)

%Kab = mean # of connections from a 'b' neuron to 'a' neurons
Kee=ceil(p(1,1)*Ne);
Kei=ceil(p(1,2)*Ne);
Kie=ceil(p(2,1)*Ni);
Kii=ceil(p(2,2)*Ni);
Kef=ceil(pf(1)*Ne);
Kif=ceil(pf(2)*Ni);

N = Ne + Ni;

ExcIds = 1:Ne;
InhIds = (Ne+1):N;

% Total number of recc/ffwd connections
Nrec = Ne * (Kee + Kie) + Ni * (Kei + Kii); % on average...
Nrec = Nrec + round(0.1 * Nrec); % for buffer room...

Nffwd = Nf * (Kef + Kif); % on average...
Nffwd = Nffwd + round(0.1 * Nffwd); % for buffer room...

% Initialize
J0 = zeros(N+1,1);
Jinds = zeros(Nrec,1);
Jweights = zeros(Nrec,1);
ii = 0; % Keeps track of the number of connections stored
inDegList = zeros(N,1);
    
% Excitatory presynaptic neurons
for j = 1:Ne
    J0(j) = ii + 1;

    %Post-synaptic 'ee' connections
    if (Kee ~= 0)
        % Get a random degree for this neuron
        numinds = binornd( Ne-1,p(1,1) );
        
        % Choose 'numinds' number of indices from pop
        postinds = randsample(ExcIds(ExcIds ~= j),numinds);
        inDegList(postinds) = inDegList(postinds) + 1;
        
        % Store indices to postsynaptic cells
        Jinds((ii+1):(ii+numinds)) = postinds;
        
        % Store synaptic weights
        Jweights((ii+1):(ii+numinds)) = J(1,1);
        
        % Update count of postsynaptic cells
        ii = ii + numinds;
    end % End ee
    
    %Post-synaptic 'ie' connections
    if (Kie ~= 0)
        % Get a random degree for this neuron
        numinds = binornd( Ni-1,p(2,1) );
        
        % Turn orientaitons into indices
        postinds = randsample(InhIds,numinds);
        inDegList(postinds) = inDegList(postinds) + 1;
        
        % Store indices to postsynaptic cells
        Jinds((ii+1):(ii+numinds)) = postinds;
        
        % Store synaptic weights
        Jweights((ii+1):(ii+numinds)) = J(2,1);
        
        % Update count of postsynaptic cells
        ii = ii + numinds;
    end % End ie
end % End e

% Inhibitory presynaptic neurons
for j = (Ne+1):N
    J0(j) = ii + 1;
    
    %Post-synaptic 'ei' connections
    if (Kei ~= 0)
        % Get a random degree for this neuron
        numinds = binornd( Ne-1,p(1,2) );
        
        % Turn orientaitons into indices
        postinds = randsample(ExcIds,numinds);
        inDegList(postinds) = inDegList(postinds) + 1;
        
        % Store indices to postsynaptic cells
        Jinds((ii+1):(ii+numinds)) = postinds;
        
        % Store synaptic weights
        Jweights((ii+1):(ii+numinds)) = J(1,2);
        
        % Update count of postsynaptic cells
        ii = ii + numinds;
    end % End ei
    
    %Post-synaptic 'ii' connections
    if (Kii ~= 0)
        % Get a random degree for this neuron
        numinds = binornd( Ni-1,p(2,2) );
        
        % Turn orientaitons into indices
        postinds = randsample(InhIds(InhIds ~= j),numinds);
        inDegList(postinds) = inDegList(postinds) + 1;
        
        % Store indices to postsynaptic cells
        Jinds((ii+1):(ii+numinds)) = postinds;
        
        % Store synaptic weights
        Jweights((ii+1):(ii+numinds)) = J(2,2);
        
        % Update count of postsynaptic cells
        ii = ii + numinds;
    end % End ii
end % End i

J0(N+1) = ii + 1;

Jinds( (ii+1):Nrec ) = []; % Clear excess pad room...
Jweights( (ii+1):Nrec ) = [];
outDegList = diff(J0);



% ----- FEEDFORWARD NETWORK ----- %

% FFwd presynaptic neurons
Jf0 = zeros(Nf+1,1);
Jfinds = zeros(Nffwd,1);
Jfweights = zeros(Nffwd,1);
ii = 0;


% FFWD presynaptic neurons
for j=1:Nf
    Jf0(j) = ii + 1;
    
    % Post-synaptic 'ef' connections
    if (Kef ~= 0)
        % Get a random degree for this neuron
        numinds = binornd( Ne-1,pf(1) );
        
        % Turn orientaitons into indices
        postinds = randsample(ExcIds,numinds);
        
        % Store indices to postsynaptic cells
        Jfinds((ii+1):(ii+numinds)) = postinds;
        
        % Store synaptic weights
        Jfweights((ii+1):(ii+numinds)) = Jf(1);
        
        % Update count of postsynaptic cells
        ii = ii + numinds;
    end % end ef
    
    %Post-synaptic 'if' connections
    if (Kif ~= 0)
        % Get a random degree for this neuron
        numinds = binornd( Ni-1,pf(2) );
        
        % Turn orientaitons into indices
        postinds = randsample(InhIds,numinds);
        
        % Store indices to postsynaptic cells
        Jfinds((ii+1):(ii+numinds)) = postinds;
        
        % Store synaptic weights
        Jfweights((ii+1):(ii+numinds)) = Jf(2);
        
        % Update count of postsynaptic cells
        ii = ii + numinds;
    end % end 'if'
end % end f

Jf0(Nf+1) = ii + 1;

Jfinds( (ii+1):Nffwd ) = []; % Clear excess pad room...
Jfweights( (ii+1):Nffwd ) = [];

end

