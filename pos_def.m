function [ X ] = pos_def( X )
%If the matrix X is not positive definite, and a multiple of the identity
%matrix (multiplied by the smallest eigenvalue) in order to render it
%positive definite

lmin = min(eig(X));
if lmin < 0
%     disp('lmin < 0')
    X = X + (-lmin + 1e-6)*eye(size(X));
end
end

