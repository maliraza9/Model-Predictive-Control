function [ x_col ] = to_column( x )
%x(:,:,n) is a sequence of vectors (or matrices) that will be placed on top
%of each other resulting in x_col

[~,~,n] = size(x);
x_col = [];
for k = 1:n
    x_col = [x_col; x(:,:,k)];
end


end

