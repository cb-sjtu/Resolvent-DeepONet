% CLENCURT nodes x (Chebyshev points) and weights w
% for Clenshaw-Curtis quadrature

% Rewriting code to make sure I know what I'm doing
% Generalized version that computes integral for arbitrary tranformation of
% physical coordinates
% here y is the regular Chebyshev colocation points in [-1,1]
% yhat is the shifted coordinate system

function [y,w] = clencurtStretch(N,dyhatdy)
theta = pi*(0:N)'/N; y = cos(theta); %y对应常规Chebyshev配置点

% keep the convention that this gives N+1 points, starting index at 0

%c0 = cn = 2, cj = 1;
% b0 = bn = 1/2, bj = 1
w = zeros(1,N+1);
Bj = ones(1,N+1);
Cn = 2*ones(1,N+1);
Bj(1) = 1/2;
Bj(end) = 1/2;
Cn(1) = 1;
Cn(end) = 1;
[~,dy] = clencurt(N); % for integration
for jj  = 0:N
    
    SumElements = 0;
    for nn = 0:N
        Tn = cos(nn*acos(y));
        Integral = dy*(Tn.*dyhatdy);
        SumElements = SumElements + Cn(nn+1)*cos(nn*jj*pi/N)*Integral;
    end
    
    w(jj+1) = Bj(jj+1)/N*(SumElements);
end

