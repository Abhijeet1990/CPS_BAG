%Example of Bayesian Network using Bayes Net toolbox
close all
clear all
clc

%Number of clusters for the Bayesian Attack Graph
Nclusters = 2;
%Number of nodes per cluster
N = 10;
%Total number of nodes in the Bayesian Attack Graph
Ntot = Nclusters*N;

sim_Count=5;
%Initialize the adjacency matrix
dag = zeros(Ntot,Ntot);

for u=2:5
%Maximum number of parents allowed per node in the BAG
max_edges = u;

for evi=1:4
sum_lbp =0;
sum_ve=0;
sum_pearl=0;
sum_jt=0;
for c=1:sim_Count
%Create the adjacency matrix
for j=1:Nclusters
    dag2 = zeros(N,N);    
    for i = 2:N
        dif = N - 1 - (N - i);
        rd = randi([1 max_edges],1,1);
        aux = 1:dif;
        ind = randperm(dif);
        aux = aux(ind);
        dag2(i, aux(1:min(rd,length(aux)))) = 1;
    end
    dag2 = dag2';
    dag(N*(j-1)+1:N*(j-1)+N,N*(j-1)+1:N*(j-1)+N) = dag2;
    c1 = 1:Ntot;
    c2 = N*(j-1)+1:N*(j-1)+N;
    c1 = setdiff(c1,c2);
    perm = randperm(length(c1));
    c1 = c1(perm);
    rd = randi([N*(j-1)+1,N*(j-1)+N]);
    if (rd > c1(1))
        dag(c1(1),rd) = 1;
    else
        dag(rd,c1(1)) = 1;
    end  
end


%All variables are Bernoulli random variables, so they have two states:
%True/False. This variable is used by BayesNet toolbox
node_sizes = 2*ones(1,Ntot); 

%Name of the nodes (in this case, to simplify, we just use the number of
%the node)
names = cell(1,Ntot);
for i=1:Ntot
    names{i} = num2str(i);
end

%Create the Bayesian network structure with Bayesnet
bnet = mk_bnet(dag, node_sizes, 'names', names, 'discrete', 1:Ntot);

%Probability of having AND-type conditional probability tables. Thus, the
%probability of having OR-type conditional probability tables is 1 - pAND
pAND = 0.2;

for i=1:Ntot
    npa = sum(dag(:,i));

    %Choose the type of conditional probability table (AND/OR) at random
    r = rand(1) > pAND;
    %Create OR conditional probability table
    if (r == 1)
        %We draw the probability from the distribution of CVSS scores
        probs = drawRandomCVSS(npa);
        cpt = createORtable(probs);
    %Create AND conditional probability table
    else
        %We draw the probability from the distribution of CVSS scores
        probs = drawRandomCVSS(npa);
        cpt = createANDtable(probs);
    end
    %Insert the conditional probability table into the Bayesnet object
    bnet.CPD{i} = tabular_CPD(bnet, i, cpt);
end


% %Show the graph
%bg = biograph(dag);
%bg.view;


    %% Approximate Inference with Loopy Belief Propagation:
    tt = cputime;
    %Engine used for Loopy Belief Propagation
    engine = belprop_fg_inf_engine(bnet_to_fgraph(bnet));
    %Vector to indicate the evidence
    evidence = cell(1,Ntot);
    %If you want to add non-empty evidence (to perform dynamic analysis), just
    if(evi==2)
        evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
    elseif(evi==3)
        evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
        evidence{2} = 2; %Set the node 1 to true (False = 1; True = 2)
    elseif(evi==4)
        evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
        evidence{2} = 2; %Set the node 1 to true (False = 1; True = 2)
        evidence{3} = 2; %Set the node 1 to true (False = 1; True = 2)
    end
    %Enter evidence
    [engine, loglik] = enter_evidence(engine, evidence);
    %Vector to store the unconditional/posterior probabilities of the nodes in
    %the Bayesian network, i.e. p(X_i = True) or p(X_i = True | Evidence)
    pTrue = zeros(Ntot,1);
    %Compute the unconditional/posterior probabilities
    for i=1:Ntot
        marg = marginal_nodes(engine, i);
        pTrue(i) = marg.T(1);    
    end
    time_loopy = cputime - tt;
    %disp('**************************************************');
    %fprintf('Inference time with LBP %1.4f\n',time_loopy);
    sum_lbp= sum_lbp + time_loopy;
    %Check if Loopy Belief propagation converged
    converged = loopy_converged(engine);
    if (converged == 0)
        fprintf('LBP did not converge\n');
    end

    %% Inference with Variable Elimination method:
    tt = cputime;
    %Engine used for Variable elimination
    engine = var_elim_inf_engine(bnet);
    %Vector to indicate the evidence
    evidence = cell(1,Ntot);
    %If you want to add non-empty evidence (to perform dynamic analysis), just
    if(evi==2)
        evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
    elseif(evi==3)
        evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
        evidence{2} = 2; %Set the node 1 to true (False = 1; True = 2)
    elseif(evi==4)
        evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
        evidence{2} = 2; %Set the node 1 to true (False = 1; True = 2)
        evidence{3} = 2; %Set the node 1 to true (False = 1; True = 2)
    end
    %Enter evidence
    [engine, loglik] = enter_evidence(engine, evidence);
    %Vector to store the unconditional/posterior probabilities of the nodes in
    %the Bayesian network, i.e. p(X_i = True) or p(X_i = True | Evidence)
    pTrueVE = zeros(Ntot,1);
    %Compute the unconditional/posterior probabilities
    for i=1:Ntot
        marg = marginal_nodes(engine, i);
        pTrueVE(i) = marg.T(1);    
    end
    time_ve = cputime - tt;
    %disp('**************************************************');
    %fprintf('Inference time with Variable Elimination %1.4f\n',time_ve);
    sum_ve = sum_ve + time_ve;

    %% Inference with Pearl Forward Backward Inference Engine
    tt = cputime;
    %Engine used for Variable elimination
    engine = pearl_inf_engine(bnet);
    %Vector to indicate the evidence
    evidence = cell(1,Ntot);
    %If you want to add non-empty evidence (to perform dynamic analysis), just
    if(evi==2)
        evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
    elseif(evi==3)
        evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
        evidence{2} = 2; %Set the node 1 to true (False = 1; True = 2)
    elseif(evi==4)
        evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
        evidence{2} = 2; %Set the node 1 to true (False = 1; True = 2)
        evidence{3} = 2; %Set the node 1 to true (False = 1; True = 2)
    end
    %Enter evidence
    [engine, loglik] = enter_evidence(engine, evidence);
    %Vector to store the unconditional/posterior probabilities of the nodes in
    %the Bayesian network, i.e. p(X_i = True) or p(X_i = True | Evidence)
    pTruePearl = zeros(Ntot,1);
    %Compute the unconditional/posterior probabilities
    for i=1:Ntot
        marg = marginal_nodes(engine, i);
        pTruePearl(i) = marg.T(1);    
    end
    time_pearl = cputime - tt;
    %disp('**************************************************');
    %fprintf('Inference time with Pearl Algo %1.4f\n',time_pearl);
    sum_pearl = sum_pearl + time_pearl;
    
%%  Inference with Junction Tree:
    tt = cputime;
    %Engine used for Junction Tree
    engine = jtree_inf_engine(bnet);
    %Vector to indicate the evidence
    evidence = cell(1,Ntot);
    %If you want to add non-empty evidence (to perform dynamic analysis), just
    if(evi==2)
        evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
    elseif(evi==3)
        evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
        evidence{2} = 2; %Set the node 1 to true (False = 1; True = 2)
    elseif(evi==4)
        evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
        evidence{2} = 2; %Set the node 1 to true (False = 1; True = 2)
        evidence{3} = 2; %Set the node 1 to true (False = 1; True = 2)
    end
    %Enter evidence
    [engine, loglik] = enter_evidence(engine, evidence);

    %Vector to store the unconditional/posterior probabilities of the nodes in
    %the Bayesian network, i.e. p(X_i = True) or p(X_i = True | Evidence)
    pTrueJT = zeros(Ntot,1);
    %Compute the unconditional/posterior probabilities
    for i=1:Ntot
        marg = marginal_nodes(engine, i);
        pTrueJT(i) = marg.T(1);    
    end

    time_jt = cputime - tt;
    %disp('**************************************************');
    %fprintf('Inference time with Junction Tree %1.4f\n',time_jt);
    sum_jt = sum_jt + time_jt;
end
fprintf('Avg. Inference time with LBP, no. of evidence %1.0f, u= %1.0f is %1.4f\n',evi-1,u,sum_lbp/sim_Count);
fprintf('Avg. Inference time with VE, no. of evidence %1.0f, u= %1.0f is %1.4f\n',evi-1,u,sum_ve/sim_Count);
fprintf('Avg. Inference time with Pearl, no. of evidence %1.0f, u= %1.0f is %1.4f\n',evi-1,u,sum_pearl/sim_Count);
fprintf('Avg. Inference time with JT, no. of evidence %1.0f, u= %1.0f is %1.4f\n',evi-1,u,sum_jt/sim_Count);
end
end
