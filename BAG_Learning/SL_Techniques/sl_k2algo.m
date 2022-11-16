% structural learning on bayesian attack graphs using K2 algorithm
close all
clear all
clc

%Number of nodes in the Bayesian Attack Graph
N =40;

% Number of simulations
sim_Count = 20;

% for all possible max_edges
for u = 2:5

avg_tt_over_all_sim = 0;
avg_acc_over_all_sim = 0;
for c = 1:sim_Count

%Maximum number of parents allowed per node in the Bayesian Attack Graph
max_edges = u;

% set some arbitrary order
%order=randperm(N,N);

order = [];
for i = 1: N
    order = [order i];
end
%Initialize adjacency matrix
dag = zeros(N,N);


%Create the adjacency matrix (at random, limiting the maximum number of
%parents per node to max_edges
for i = 2:N
    dif = N - 1 - (N - i);
    rd = randi([1 max_edges],1,1);
    aux = 1:dif;
    ind = randperm(dif);
    aux = aux(ind);
    dag(i, aux(1:min(rd,length(aux)))) = 1;
end
dag = dag';

%All variables are Bernoulli random variables, so they have two states:
%True/False. This variable is used by BayesNet toolbox
node_sizes = 2*ones(1,N); 

%Name of the nodes (in this case, to simplify, we just use the number of
%the node)
names = cell(1,N);
for i=1:N
    names{i} = num2str(i);
end

%Create the Bayesian network structure with Bayesnet
bnet = mk_bnet(dag, node_sizes, 'names', names, 'discrete', 1:N);

%Probability of having AND-type conditional probability tables. Thus, the
%probability of having OR-type conditional probability tables is 1 - pAND
pAND = 0.2;

for i=1:N
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
%Show the Bayesian network (plot)
bg = biograph(dag);
%bg.view;
%bnet.CPD

seed = 0;
rand('state', seed);
randn('state', seed);
ncases = 1000;
data = zeros(N, ncases);
for m=1:ncases
  data(:,m) = cell2num(sample_bnet(bnet));
end


max_fan_in = max_edges;

%dag2 = learn_struct_K2(data, ns, order, 'max_fan_in', max_fan_in, 'verbose', 'yes');
dagK2=dag;  
sz = 100:100:1000;
for i=1:length(sz)
  dag2 = learn_struct_K2(data(:,1:sz(i)), node_sizes, order, 'max_fan_in', max_fan_in);
  correct(i) = isequal(dag, dag2);
%   bg = biograph(dag2);
%   bg.view;
end
%correct
tt_k2=0;
sum_tt=0;
sum_acc=0;

for i=1:length(sz)

  tt_k2=cputime;

  dag3 = learn_struct_K2(data(:,1:sz(i)), node_sizes, order, 'max_fan_in', max_fan_in, 'scoring_fn', 'bic', 'params', []);
  correct(i) = isequal(dag, dag3);
  dagK2 = dag3;
  acc = accuracy(dag,dag3);
  time_k2= cputime-tt_k2;
  sum_tt = sum_tt + time_k2;
  sum_acc = sum_acc + acc;
  %disp('**************************************************');
  %fprintf('Learning time with K2 algorihtm %1.4f with accuracy %1.4f\n',time_k2,acc);

end
%fprintf('Average Learning time with K2 algorihtm %1.4f and accuracy %1.4f \n',sum_tt/10, sum_acc/10);
avg_tt_over_all_sim =avg_tt_over_all_sim + sum_tt/10;
avg_acc_over_all_sim = avg_acc_over_all_sim+ sum_acc/10;
end
fprintf('Average Learning time with K2 algorihtm %1.4f and accuracy %1.4f  over all sim \n',avg_tt_over_all_sim/sim_Count, avg_acc_over_all_sim/sim_Count);
end

function acc = accuracy(dag,dag1)
match = 0;
for i = 1:length(dag)
    for j = 1:length(dag)
        if dag(i,j) == dag1(i,j)
            match=match+1;
        end
    end
end
acc = match/(length(dag)*length(dag));

end
