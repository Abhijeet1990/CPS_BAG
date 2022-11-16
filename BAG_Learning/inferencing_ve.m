%Example of Bayesian Network using Bayes Net toolbox
close all
clear all
clc
% Number of simulations
sim_Count = 20;
%Number of nodes in the Bayesian Attack Graph
for evi=1:2
 
 for N = 6:2:14
    %Initialize adjacency matrix
    dag = zeros(N,N);
    %Maximum number of parents allowed per node in the Bayesian Attack Graph
    for u = 2:1:5
       avg_tt_over_all_sim = 0;
       

       for c = 1:sim_Count   
        %Initialize adjacency matrix
        dag = zeros(N,N);
        max_edges = u;
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
        tt = cputime;
        %Engine used for Variable Elimination method
        engine = var_elim_inf_engine(bnet);
        %Vector to indicate the evidence
        evidence = cell(1,N);
        %If you want to add non-empty evidence (to perform dynamic analysis), just
        if(evi==2)
            evidence{1} = 2; %Set the node 1 to true (False = 1; True = 2)
        end
        %Enter evidence
        [engine, loglik] = enter_evidence(engine, evidence);

        %Vector to store the unconditional/posterior probabilities of the nodes in
        %the Bayesian network, i.e. p(X_i = True) or p(X_i = True | Evidence)
        pTrueVE = zeros(N,1);
        %Compute the unconditional/posterior probabilities
        for i=1:N
            marg = marginal_nodes(engine, i);
            pTrueVE(i) = marg.T(1);    
        end
%         time_jt = cputime - tt;
%         disp('**************************************************');
%         fprintf('Inference time with Junction Tree %1.4f\n',time_jt);
        avg_tt_over_all_sim = avg_tt_over_all_sim + (cputime - tt);
       end
       fprintf('Average Inference time %1.4f over all sim for evidence %1.0f u=%1.0f Nodes= %1.0f\n',avg_tt_over_all_sim/sim_Count, evi,u,N);
     end
  end
end