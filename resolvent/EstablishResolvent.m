function [u0, s0, v0, SumSvalSq,s0all] = EstablishResolvent(Ret,kx,kz,cP,N,nsvd,Um,U_prime,U_2prime)
% establish the resolvent operator
    omega = kx* cP;
    [y,DM] = chebdif(N,2); 
    D1=DM(:,:,1);
    D2=DM(:,:,2);

    [~,dy] = clencurt(N-1);

    I = eye(N); Z = zeros(N);
    Delta = D2-(kx^2+kz^2)*I;
    
    M = [I Z Z Z; Z I Z Z; Z Z I Z; Z Z Z Z];

    L11 = 1i*kx*Um-1/Ret*Delta;
    L12 = U_prime;
    L13 = Z;
    L14 = 1i*kx*I;
    L21 = Z; 
    L22 = 1i*kx*Um-1/Ret*Delta;
    L23 = Z;
    L24 = D1;
    L31 = Z;
    L32 = Z;
    L33 = 1i*kx*Um-1/Ret*Delta;
    L34 = 1i*kz*I;
    L41 = 1i*kx*I;
    L42 = D1;
    L43 = 1i*kz*I;
    L44 = Z;

    L = [L11 L12 L13 L14;
        L21 L22 L23 L24;
        L31 L32 L33 L34;
        L41 L42 L43 L44];


    LHS = -1i*omega*M+L;
    RHS = [I Z Z; Z I Z; Z Z I; Z Z Z];

    H0 = BC(LHS,RHS,N,D1);

    IW = sqrtm(diag(dy)); 
    sqWl = [IW Z Z Z; Z IW Z Z; Z Z IW Z; Z Z Z Z];
    sqWr = [IW Z Z; Z IW Z; Z Z IW];
    H0W = sqWl*H0/sqWr;
 

    [u0W,s0W,v0W] = svd(H0W);
    SumSvalSq = sum(diag(s0W).^2);
    s0all = diag(s0W);
    u0W = u0W(:,1:nsvd);
    s0W = s0W(1:nsvd,1:nsvd);
    v0W = v0W(:,1:nsvd);
    
    u0 = pinv(sqWl)*u0W;
    v0 = sqWr\v0W;
    s0 = diag(s0W);

    if(cP < max(diag(Um)))
        ind = find(flipud(diag(Um))>cP,1,'first');
        ind = N-ind+1;
    else
        ind = 1;
    end
    phase_shift = -1i*angle(u0(ind,:));
    v0 = v0*diag(exp(phase_shift));
    u0 = H0*v0;
    
    u0 = u0*diag(1./s0);
    Um = diag(Um);
end
