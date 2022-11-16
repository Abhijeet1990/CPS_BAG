function [cpt] = createANDtable(probs)

if (isempty(probs))
    %cpt = [0 1];
    cpt = [0.2 0.8];
else
    npa = length(probs);
    cpt = zeros(2, 2^npa);
    cpt(2,end) = prod(probs);
    cpt(1,:) = 1 - cpt(2,:);
end

cpt = cpt';

end
