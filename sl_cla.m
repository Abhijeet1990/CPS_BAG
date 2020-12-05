% structural learning on bayesian attack graphs
close all
clear all
clc

sim_Count = 5;
%Number of nodes in the Bayesian Attack Graph
for N = 12:8:20
%Maximum number of parents allowed per node in the Bayesian Attack Graph
for u = 2:1:5
max_edges = u;
avg_tt_over_all_sim = 0;
avg_acc_over_all_sim = 0;
for c = 1:sim_Count
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

ncases = 1000;
data = zeros(N, ncases);
for m=1:ncases
  data(:,m) = cell2num(sample_bnet(bnet));
end

%data_new = int2str(data');
data_new = convertIntToChar(data');
tt = cputime;
tree = createChouLiuTree(data_new);
time_cla = cputime - tt;
mst_edges = tree.Edges;
end_Nodes = mst_edges.EndNodes;

dag_cla = zeros(N,N);
for i = 1:length(end_Nodes)
    dag_cla(end_Nodes(i,1),end_Nodes(i,2)) = 1;
end

avg_tt_over_all_sim = avg_tt_over_all_sim + time_cla;
avg_acc_over_all_sim = avg_acc_over_all_sim + accuracy(dag,dag_cla);
%fprintf('Accuracy %1.4f',accuracy(dag,dag_cla));

end
fprintf('Chou Liu algorihtm N=%1.0f u=%1.0f Average comp time:  %1.4f and avg. accuracy: %1.4f\n',N,u,avg_tt_over_all_sim/sim_Count, avg_acc_over_all_sim/sim_Count);
end
end


% convert integer matrix to character matrix
function char_mat = convertIntToChar(data)
    char_mat = repmat(' ',[size(data,1) size(data,2)]);
    for i = 1:size(data,1)
        for j = 1:size(data,2)
            char_mat(i,j) = num2str(data(i,j));
        end
    end
end

% check for accuracy
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

% create Chou Liu Tree
function T = createChouLiuTree(data)
    G = graph;
    for v= 2:size(data,2)
        G = addnode(G,v);
        for u=1:v
            G = addedge(G,u,v,-get_mutual_information(data,u,v));
        end
    end
    T = minspantree(G);
end

% get mutual information
function I = get_mutual_information(data, u, v)
    if(u > v)
        t = u;
        u = v;
        v = t;
    end
    marg_u  = marginal_dist(data,u); 
    mu_keys = keys(marg_u); 
    marg_v = marginal_dist(data,v); 
    mv_keys = keys(marg_v); 
    marg_uv = marginal_pair_dist(data,u,v);

    I = 0;    
    for i = 1:length(mu_keys)
        for j = 1:length(mv_keys)
            if isKey(marg_uv,strcat(mu_keys(i),mv_keys(j)))
                p_data_uv = values(marg_uv,strcat(mu_keys(i),mv_keys(j)));
                p_data_u = values(marg_u,mu_keys(i));
                p_data_v = values(marg_v,mv_keys(j));
                I = I + cell2mat(p_data_uv) * (log(cell2mat(p_data_uv)) - log(cell2mat(p_data_u)) - log(cell2mat(p_data_v)));
            end
        end
    end    
end

% marginal pair distribution
function values = marginal_pair_dist(data, u, v)
    if(u > v)
        t = u;
        u = v;
        v = t;
    end
    values = containers.Map;
    s = 1/size(data,1);
    for i =1:size(data,1)
        values(strcat(data(i,u),data(i,v))) = 0;
    end  
    for i =1:size(data,1)
        values(strcat(data(i,u),data(i,v))) = values(strcat(data(i,u),data(i,v)))+s;
    end 
end

% marginal distribution of each feature
function values = marginal_dist(data, u)
    values = containers.Map;
    s = 1/size(data,1);
    for i =1:size(data,1)
        values(data(i,u)) = 0;
    end  
    for i =1:size(data,1)
        values(data(i,u)) = values(data(i,u))+s;
    end 
end




