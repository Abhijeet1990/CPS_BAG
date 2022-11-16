%Example of Bayesian Network using Bayes Net toolbox
close all
clear all
clc
% Number of simulations
sim_Count = 20;

% say we have one utility CC and 3 substation model
N_sub = 3
util_Node = ["util_ba","ba_router","ba_firewall","util_iccp","util_switch","dnp_master","HMI", "ba_sub_firewall","ba_sub_router","pub_switch","pub_db","pub_ws","dmz_firewall","vendor_switch","corp_switch","vendor_node","corp_node"]
sub_Node = ["sub_router","sub_firewall", "sub_switch","local_db","local_ws","sub_ot_switch","sub_rtac","sub_rel1","sub_rel2","sub_rel3","sub_rel4"]

function dag = create_util(dag,sub_Node,N,N_sub)
	%Initialize adjacency matrix
	dag = zeros(N,N);
	dag(0,1) = 1; 
	dag(1,2) = 1;
	dag(2,3) = 1;
	dag(2,4) = 1;
	dag(4,5) = 1;
	dag(4,6) = 1;
	dag(4,7) = 1;
	dag(7,8) = 1; % 8 is ba_Sub_router
	dag(7,9) = 1;
	dag(9,10) = 1;
	dag(9,11) = 1;
	dag(9,12) = 1;
	dag(12,13) = 1;
	dag(12,14) = 1;
	dag(13,15) = 1;
	dag(14,16) = 1;
	
	% add the links to the substation router
	for k = 1:N_sub
		dag(8,17+size(sub_Node)*(k-1)) = 1;
	end
end
	
function dag = create_subs(dag,sub_Node,N_sub)
	for k = 1:N_sub
		j = 17+size(sub_Node)*(k-1);
		dag(j,j+1) = 1;
		dag(j+1,j+2) = 1;
		dag(j+1,j+3) = 1;
		dag(j+1,j+4) = 1;
		dag(j+4,j+5) = 1;
		dag(j+4,j+6) = 1;
		dag(j+4,j+7) = 1;
		dag(j+4,j+8) = 1;
		dag(j+4,j+9) = 1;
	end
end

%Number of nodes in the Bayesian Attack Graph
for evi=1:2
	N = size(util_Node) + N_sub*size(sub_Node)
    %Initialize adjacency matrix
    dag = zeros(N,N);

       avg_tt_over_all_sim = 0;
       

       for c = 1:sim_Count   
        
        
		% create a DAG based on the n/w model
		dag = create_util(dag,sub_Node,N,N_sub);
		dag = create_subs(dag,sub_Node,N_sub);
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