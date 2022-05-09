function [ x, p, VN, grad_VN ] = fb_sweep( x, u, funcs )
%forward backward sweep to calculate the states x, costates p, cost VN and 
%its gradient with respect to u


ell = funcs.ell;
grad_ell_u = funcs.grad_ell_u;
grad_ell_x = funcs.grad_ell_x;
Vf = funcs.Vf;
grad_Vf = funcs.grad_Vf;

f = funcs.f;
grad_f_u = funcs.grad_f_u;
grad_f_x = funcs.grad_f_x;

N = size(u,3);
p = zeros(size(x,1),1,N);
VN = 0;
grad_VN = [];

for k = 1:N
    VN = VN + ell(x(:,:,k),u(:,:,k));
    x(:,:,k+1) = full(f(x(:,:,k), u(:,:,k)));
end
% x(:,:,N+1)
VN = VN + Vf(x(:,:,N+1));

if (nargout == 3) %this means only the cost VN is required 
    return;
end
    
p(:,:,N) = grad_Vf(x(:,:,N+1));
for k = N:-1:2
    grad_VN_uk = grad_ell_u(x(:,:,k),u(:,:,k)) + grad_f_u(x(:,:,k),u(:,:,k))*p(:,:,k);
    grad_VN = [grad_VN_uk; grad_VN];
    p(:,:,k-1) = grad_ell_x(x(:,:,k),u(:,:,k)) + grad_f_x(x(:,:,k),u(:,:,k))*p(:,:,k);
end
grad_VN_uk = grad_ell_u(x(:,:,1),u(:,:,1)) + grad_f_u(x(:,:,1),u(:,:,1))*p(:,:,1);
grad_VN = [grad_VN_uk; grad_VN];
end

