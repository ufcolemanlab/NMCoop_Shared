function smoothed_z = EWMA(z,L)
%
% Computes the exponentially weighted moving average (with memory L) of
% input data z
%
% 9/24/16 - Removed 'i' loop since we are only using a vector/1D array for z

lambda = 1-2/(L+1);

smoothed_z = zeros(size(z));

smoothed_z = z;

for j = 2:size(z,2)
    smoothed_z(j) = lambda * smoothed_z(j-1) + (1-lambda) * z(j);
end
