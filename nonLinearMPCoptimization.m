close all
clear all
clc

import casadi.*

%% Exercise 1: Casadi and computing Hessians
N = 20;
nx = 3;
nu = 1;

Q = eye(3);
R = 0.01;
P = 5*Q;

xk = SX.sym('xk', nx);
uk = SX.sym('uk', nu);
xk1 = SX.sym('xk1', nx); %x_k+1
pk = SX.sym('pk', nx);

sys_dyns = [10*(xk(2)-xk(1)); 
               xk(1)*(uk - xk(3)) - xk(2);
               xk(1)*xk(2) - 3*xk(3)];
           
           
dt = 0.05;
f = xk + dt*sys_dyns; %forward euler

% validation: for a linear system the Newton methods should converge in one
% iteration

% A = [1 0.5 0.01; 1 0 0; 0 1 0];
% B = [1; 0.5; 0.1];
% f = A*xk + B*uk;
           
ell = 1/2*xk'*Q*xk + 1/2*uk'*R*uk;
Vf = 1/2*xk'*P*xk;

H = ell + pk'*f;

Qk = casadi.Function('Qk',{xk, uk, pk}, {hessian(H,xk)});
Rk = casadi.Function('Rk',{xk, uk, pk}, {hessian(H,uk)});
Sk = casadi.Function('Sk',{xk, uk, pk}, {jacobian(gradient(H,uk),xk)});
qN = casadi.Function('qN',{xk}, {gradient(Vf,xk)});
qk = casadi.Function('qk',{xk, uk}, {gradient(ell,xk)});
rk = casadi.Function('rk',{xk, uk}, {gradient(ell,uk)});
rk2 = casadi.Function('rk2',{xk, uk, pk}, {gradient(H,uk)}); %= rk for the sequential approach
QN = casadi.Function('QN',{xk}, {hessian(Vf,xk)});
Ak = casadi.Function('Ak',{xk, uk}, {jacobian(f,xk)});
Bk = casadi.Function('Bk',{xk, uk}, {jacobian(f,uk)});
ck = casadi.Function('ck',{xk, uk, xk1}, {f - xk1});
f = casadi.Function('f',{xk, uk}, {f});

x = SX.sym('x',nx*(N+1));
u = SX.sym('u',nu*N);
c = SX.sym('c',1);

JN = 0;
h = 0;
for k = 0:N-1
    JN = JN + 1/2*x(k*nx+1:(k+1)*nx)'*Q*x(k*nx+1:(k+1)*nx) + 1/2*u(k*nu+1:(k+1)*nu)'*R*u(k*nu+1:(k+1)*nu);
    h = h + norm(abs(f(x(k*nx+1:(k+1)*nx), u(k*nu+1:(k+1)*nu)) - x((k+1)*nx+1:(k+2)*nx)));
end
JN = JN + 1/2*x(N*nx+1:(N+1)*nx)'*P*x(N*nx+1:(N+1)*nx);

phi = casadi.Function('phi',{c, x, u}, {JN + c*h});
jac_JN_x = casadi.Function('jac_JN_x',{x, u},{jacobian(JN,x)});
jac_JN_u = casadi.Function('jac_JN_u',{x, u},{jacobian(JN,u)});
h = casadi.Function('h',{x,u},{h});
JN = casadi.Function('JN',{x,u},{JN});

%% Exercise 2: Newton-Lagrange: LQR step

% % %initial guess
fprintf('Newton Lagrange\n\n')
u = zeros(nu,1,N);
x = zeros(nx,1,N+1);
x(:,:,1) = [2;15;10]; %initial state
for k = 1:N
    x(:,:,k+1) = full(f(x(:,:,k), u(:,:,k)));
end

p = zeros(nx,1,N);

maxiter = 20;
for i = 1:maxiter

    fprintf('JN: %f, ||h||: %f\n',full(JN(to_column(x),to_column(u))),full(h(to_column(x),to_column(u))))
    
    %LQR factor step
    K = zeros(nu,nx,N);
    s = zeros(nx,1,N);
    P = zeros(nx,nx,N);
    e = zeros(nu,1,N);
    P(:,:,N) = full(QN(x(:,:,N+1)));
    s(:,:,N) = full(qN(x(:,:,N+1)));
    
    %Compute and store some matrices/vectors from casadi
    for k = 1:N
        
        xk = x(:,:,k);
        uk = u(:,:,k);
        xk1 = x(:,:,k+1);
        pk = p(:,:,k);
        
        Aki(:,:,k) = full(Ak(xk, uk));
        Bki(:,:,k) = full(Bk(xk, uk));
        cki(:,:,k) = full(ck(xk, uk, xk1));
        Rki(:,:,k) = full(Rk(xk, uk, pk));
        Qki(:,:,k) = full(Qk(xk, uk, pk));
        Ski(:,:,k) = full(Sk(xk, uk, pk));
        rki(:,:,k) = full(rk(xk, uk));
        qki(:,:,k) = full(qk(xk, uk));
        
    end
    
    for k = N-1:-1:0 %Note that the indices have to move one because matlab starts counting from 1
        
        R_bar = Rki(:,:,k+1) + Bki(:,:,k+1)'*P(:,:,k+1)*Bki(:,:,k+1);
        S_bar = Ski(:,:,k+1) + Bki(:,:,k+1)'*P(:,:,k+1)*Aki(:,:,k+1);
        y = P(:,:,k+1)*cki(:,:,k+1) + s(:,:,k+1);
        K(:,:,k+1) = -R_bar\S_bar;
        e(:,:,k+1) = -R_bar\(Bki(:,:,k+1)'*y + rki(:,:,k+1));
        if k>0
            s(:,:,k) = S_bar'*e(:,:,k+1) + Aki(:,:,k+1)'*y + qki(:,:,k+1);
            P(:,:,k) = Qki(:,:,k+1) + Aki(:,:,k+1)'*P(:,:,k+1)*Aki(:,:,k+1) + S_bar'*K(:,:,k+1);
%             P(:,:,k) = pos_def(P(:,:,k));
        end
        
    end

    %LQR solve step
    delta_x = zeros(nx,1,N+1);
    delta_u = zeros(nu,1,N);

    for k = 1:N
        xk = x(:,:,k);
        uk = u(:,:,k);
        xk1 = x(:,:,k+1);
        delta_u(:,:,k) = K(:,:,k)*delta_x(:,:,k) + e(:,:,k);
        delta_x(:,:,k+1) = Aki(:,:,k)*delta_x(:,:,k) + Bki(:,:,k)*delta_u(:,:,k) + cki(:,:,k);
        p(:,:,k) = P(:,:,k)*delta_x(:,:,k+1) + s(:,:,k);
    end

    %Update inputs and states
    for k=1:N
        u(:,:,k) = u(:,:,k) + delta_u(:,:,k);
        x(:,:,k+1) = x(:,:,k+1) + delta_x(:,:,k+1);
    end
    
    if norm(to_column(delta_u), Inf)/norm(to_column(u)) < 1e-6
        disp('No more progress')
        break;
    end
   
    
end

%% Exercise 3: Linesearch on merit function
% 
fprintf('\n\nNewton-Lagrange with linesearch\n\n')
sigma = 1e-4;

%initial guess
u = zeros(nu,1,N);
x = zeros(nx,1,N+1);
x(:,:,1) = [2;15;10]; %initial state
for k = 1:N
    x(:,:,k+1) = full(f(x(:,:,k), u(:,:,k)));
end

p = zeros(nx,1,N);

maxiter = 20;
for i = 1:maxiter

     %LQR factor step
    K = zeros(nu,nx,N);
    s = zeros(nx,1,N);
    P = zeros(nx,nx,N);
    e = zeros(nu,1,N);
    P(:,:,N) = full(QN(x(:,:,N+1)));
    s(:,:,N) = full(qN(x(:,:,N+1)));
    
    %Compute and store some matrices/vectors from casadi
    for k = 1:N
        
        xk = x(:,:,k);
        uk = u(:,:,k);
        xk1 = x(:,:,k+1);
        pk = p(:,:,k);
        
        Aki(:,:,k) = full(Ak(xk, uk));
        Bki(:,:,k) = full(Bk(xk, uk));
        cki(:,:,k) = full(ck(xk, uk, xk1));
        Rki(:,:,k) = full(Rk(xk, uk, pk));
        Qki(:,:,k) = full(Qk(xk, uk, pk));
        Ski(:,:,k) = full(Sk(xk, uk, pk));
        rki(:,:,k) = full(rk(xk, uk));
        qki(:,:,k) = full(qk(xk, uk));
        
    end
    
    for k = N-1:-1:0 %Note that the indices have to move one because matlab starts counting from 1
        
        R_bar = Rki(:,:,k+1) + Bki(:,:,k+1)'*P(:,:,k+1)*Bki(:,:,k+1);
        S_bar = Ski(:,:,k+1) + Bki(:,:,k+1)'*P(:,:,k+1)*Aki(:,:,k+1);
        y = P(:,:,k+1)*cki(:,:,k+1) + s(:,:,k+1);
        K(:,:,k+1) = -R_bar\S_bar;
        e(:,:,k+1) = -R_bar\(Bki(:,:,k+1)'*y + rki(:,:,k+1));
        if k>0
            s(:,:,k) = S_bar'*e(:,:,k+1) + Aki(:,:,k+1)'*y + qki(:,:,k+1);
            P(:,:,k) = Qki(:,:,k+1) + Aki(:,:,k+1)'*P(:,:,k+1)*Aki(:,:,k+1) + S_bar'*K(:,:,k+1);
%             P(:,:,k) = pos_def(P(:,:,k));
        end
        
    end
    
    %LQR solve step
    delta_x = zeros(nx,1,N+1);
    delta_u = zeros(nu,1,N);

    for k = 1:N
        xk = x(:,:,k);
        uk = u(:,:,k);
        xk1 = x(:,:,k+1);
        delta_u(:,:,k) = K(:,:,k)*delta_x(:,:,k) + e(:,:,k);
        delta_x(:,:,k+1) = Aki(:,:,k)*delta_x(:,:,k) + Bki(:,:,k)*delta_u(:,:,k) + cki(:,:,k);
        p(:,:,k) = P(:,:,k)*delta_x(:,:,k+1) + s(:,:,k);
    end

    %Update inputs and states with linesearch
    alpha = 1;
    c = 0;
    for i = 1:N 
        c = max(c,norm(p(:,:,i),Inf));
    end
    
    delta_x_col = to_column(delta_x);
    delta_u_col = to_column(delta_u);
    x_col = to_column(x);
    u_col = to_column(u);
    
    while 1
        
        xplus(:,:,1) = x(:,:,1);
        for k=1:N
            uplus(:,:,k) = u(:,:,k) + alpha*delta_u(:,:,k);
            xplus(:,:,k+1) = x(:,:,k+1) + alpha*delta_x(:,:,k+1);
        end

        xplus_col = to_column(xplus);
        uplus_col = to_column(uplus);

        if full(phi(c,xplus_col,uplus_col) <= phi(c,x_col,u_col) + ...
                sigma*alpha*(full(jac_JN_x(x_col,u_col))*delta_x_col + full(jac_JN_u(x_col,u_col))*delta_u_col - c*h(x_col,u_col)))
            u = uplus;
            x = xplus;
            break;
        end
        
        alpha = alpha/2;
        
    end
    
    fprintf('JN: %f, ||h||: %f, c: %f \n',full(JN(x_col,u_col)),full(h(x_col,u_col)), c)
    
    if norm(to_column(delta_u), Inf)/norm(to_column(u)) < 1e-6
        disp('No more progress')
        break;
    end
    
end


%% Exercise 4: Sequential approach: gradient method

fprintf('\n\n Gradient method for sequential approach\n\n')

sigma = 1e-4;

u = zeros(nu,1,N);
x = zeros(nx,1,N+1);
x(:,:,1) = [2;15;10]; %initial state
for k = 1:N
    x(:,:,k+1) = full(f(x(:,:,k), u(:,:,k)));
end

Q = eye(3);
R = 0.01;
P = 5*Q;

funcs.ell = @(x,u) 1/2*(x'*Q*x + u'*R*u);
funcs.grad_ell_u = @(x,u) R*u;
funcs.grad_ell_x = @(x,u) Q*x;
funcs.Vf = @(x) 1/2*x'*P*x;
funcs.grad_Vf = @(x) P*x;

funcs.f = @(x,u) full(f(x,u));
funcs.grad_f_u = @(x,u) full(Bk(x,u))';
funcs.grad_f_x = @(x,u) full(Ak(x,u))';

maxiter = 400;
for i = 1:maxiter
    %Compute states x and cost and gradient by forward-backward sweep
    [x, ~, VN, grad_VN] = fb_sweep(x, u, funcs);
    %update inputs
    alpha = 1;
    
    %delta_u is the steepest descent step
    for k = 1:N
        delta_u(:,:,k) = -grad_VN((k-1)*nu+1:k*nu);
    end
    
    while 1
        for k = 1:N
            uplus(:,:,k) = u(:,:,k) + alpha*delta_u(:,:,k);
        end
        [~,~,VN_uplus] = fb_sweep(x,uplus,funcs);
        if (VN_uplus <= VN + sigma*alpha*grad_VN'*to_column(delta_u)) 
            u = uplus;          
            break;
        end
        alpha = alpha/10;
    end
    
    if norm(alpha*to_column(delta_u), Inf)/norm(to_column(u)) < 1e-6
        disp('No more progress')
        break;
    end
        
    fprintf('VN: %f, ||grad_VN||: %f, alpha: %f\n', VN, norm(grad_VN), alpha);
end

%% Exercise 5: Sequential approach: Newton method
% pause
fprintf('\n\nNewton method for sequential approach\n\n')
sigma = 1e-4;


u = zeros(nu,1,N);
x = zeros(nx,1,N+1);
x(:,:,1) = [2;15;10]; %initial state
for k = 1:N
    x(:,:,k+1) = full(f(x(:,:,k), u(:,:,k)));
end

p = zeros(nx,1,N);

Q = eye(3);
R = 0.01;
P = 5*Q;

funcs.ell = @(x,u) 1/2*(x'*Q*x + u'*R*u);
funcs.grad_ell_u = @(x,u) R*u;
funcs.grad_ell_x = @(x,u) Q*x;
funcs.Vf = @(x) 1/2*x'*P*x;
funcs.grad_Vf = @(x) P*x;

funcs.f = @(x,u) full(f(x,u));
funcs.grad_f_u = @(x,u) full(Bk(x,u))';
funcs.grad_f_x = @(x,u) full(Ak(x,u))';

maxiter = 20;
for i = 1:maxiter
    %Compute states x and costates p and gradient by forward-backward sweep
    [x, p, VN, grad_VN] = fb_sweep(x, u, funcs);


    %LQR factor step
    K = zeros(nu,nx,N);
    s = zeros(nx,1,N);
    P = zeros(nx,nx,N);
    e = zeros(nu,1,N);
    P(:,:,N) = full(QN(x(:,:,N+1)));
    s(:,:,N) = zeros(nx,1);
    
    for k = 1:N
        
        xk = x(:,:,k);
        uk = u(:,:,k);
        xk1 = x(:,:,k+1);
        pk = p(:,:,k);
        
        Aki(:,:,k) = full(Ak(xk, uk));
        Bki(:,:,k) = full(Bk(xk, uk));
        Rki(:,:,k) = full(Rk(xk, uk, pk));
        Qki(:,:,k) = full(Qk(xk, uk, pk));
        Ski(:,:,k) = full(Sk(xk, uk, pk));
        rk2i(:,:,k) = full(rk2(xk, uk, pk));
       
    end
    
    
    for k = N-1:-1:0
        R_bar = Rki(:,:,k+1) + Bki(:,:,k+1)'*P(:,:,k+1)*Bki(:,:,k+1);
        S_bar = Ski(:,:,k+1) + Bki(:,:,k+1)'*P(:,:,k+1)*Aki(:,:,k+1);
        y = s(:,:,k+1);
        K(:,:,k+1) = -R_bar\S_bar;
        e(:,:,k+1) = -R_bar\(Bki(:,:,k+1)'*y + rk2i(:,:,k+1));
        if k>0
            s(:,:,k) = S_bar'*e(:,:,k+1) + Aki(:,:,k+1)'*y;
            P(:,:,k) = Qki(:,:,k+1) + Aki(:,:,k+1)'*P(:,:,k+1)*Aki(:,:,k+1) + S_bar'*K(:,:,k+1);
            P(:,:,k) = pos_def(P(:,:,k));
        end
    end

    %Calculate delta_u
    delta_x = zeros(nx,1,N+1);
    delta_u = zeros(nu,1,N);

    for k = 1:N
        xk = x(:,:,k);
        uk = u(:,:,k);
        delta_u(:,:,k) = K(:,:,k)*delta_x(:,:,k) + e(:,:,k);
        delta_x(:,:,k+1) = Aki(:,:,k)*delta_x(:,:,k) + Bki(:,:,k)*delta_u(:,:,k);
    end

    %update inputs
    alpha = 1;
    while 1
        for k = 1:N
            uplus(:,:,k) = u(:,:,k) + alpha*delta_u(:,:,k);
        end
        [~,~,VN_uplus] = fb_sweep(x,uplus,funcs);
        if (VN_uplus <= VN + sigma*alpha*grad_VN'*to_column(delta_u))
            u = uplus;
            
            break;
        end
        alpha = alpha/10;
    end
    
    fprintf('VN: %f, ||grad_VN||: %f, alpha: %e\n', VN, norm(grad_VN), alpha);
    
%     if norm(alpha*to_column(delta_u), Inf)/norm(to_column(u)) < 1e-6
%         disp('No more progress')
%         break;
%     end
end


