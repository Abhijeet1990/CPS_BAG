function [probs] = drawRandomCVSS(n)

perc = [0.10 0.70 4.10 2.30 19.40 20.70 11.90 26.40 0.40 14.00]./100;
vals = [0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95];

probs = zeros(n,1);

for i=1:n
    r = mnrnd(1,perc);
    c = find(r == 1);
    probs(i) = vals(c);
end


%Distribution of CVSS scores
% 0-1   50      0.10
% 1-2 	518 	0.70
% 2-3 	2855 	4.10
% 3-4 	1565 	2.30
% 4-5 	13469 	19.40
% 5-6 	14320 	20.70
% 6-7 	8218 	11.90
% 7-8 	18258 	26.40
% 8-9 	278 	0.40
% 9-10 	9721 	14.00 


end

