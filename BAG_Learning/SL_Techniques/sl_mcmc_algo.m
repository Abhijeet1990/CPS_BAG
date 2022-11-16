% structural learning on bayesian attack graphs using MCMC algorithm
close all
clear all
clc

%Number of nodes in the Bayesian Attack Graph
N =10;

% Number of simulations
sim_Count = 20;

% for all possible max_edges
for u = 2:5
for ncases = 1000:1000:5000
    avg_tt_over_all_sim = 0;
    avg_acc_over_all_sim = 0;
    for c = 1:sim_Count

        %Maximum number of parents allowed per node in the Bayesian Attack Graph
        max_edges = u;

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
        % Testing MCMC structural learning

        data = zeros(N, ncases);
        for m=1:ncases
            data(:,m) = cell2num(sample_bnet(bnet));
        end
        tt_mcmc = cputime;
        [sampled_graphs, accept_ratio] = learn_struct_mcmc(data, node_sizes, 'nsamples',ncases, 'burnin', 10);
        %accept_ratio
        acc = accuracy(dag,sampled_graphs{ncases});
        time_mcmc = cputime - tt_mcmc;
        %disp('**************************************************');
        %fprintf('Learning time with MCMC %1.4f using %1.0f samples \n',time_mcmc,ncases);
        avg_tt_over_all_sim =avg_tt_over_all_sim + time_mcmc;
        avg_acc_over_all_sim = avg_acc_over_all_sim+ acc;
    end
    fprintf('Average Learning time MCMC %1.4f over all sim %1.1f cases, u = %1.1f accuracy = %1.4f\n',avg_tt_over_all_sim/sim_Count,ncases,u,avg_acc_over_all_sim/sim_Count);
end
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
