% boundary condition for channel flows
function H=BC(LHS,RHS,N,D1)

% no-slip condition
for ni = [1,N,N+1,2*N,2*N+1,3*N] 
    LHS(ni,:)=0;
    RHS(ni,:)=0;
    LHS(ni,ni)=1;
end

% no-penetration condition
% LHS(1,1:N)=D1(1,:);
% LHS(2*N,N+1:2*N)=D1(N,:); % at the wall

%HH = LHS\RHS;
if rank(LHS)<4*N
    disp('not full rank');
    disp(rank(LHS));
    H = pinv(LHS)*RHS;
else
    H = LHS\RHS;
end

% H(N,:) = 0.;
% H(2*N,:) = 0.;
% H(3*N,:) = 0.;
% H(1,:) = 0.;
% % H(N+1,:) = 0.;
% H(2*N+1,:) = 0.;


end


