function dfdx=diff2(f,x,nx)

for ii=1:nx
    if ii==1
        dfdx(ii)=(f(ii+1)-f(ii))/(x(ii+1)-x(ii));
    elseif ii==nx
        dfdx(ii)=(f(ii)-f(ii-1))/(x(ii)-x(ii-1));
    else
        dfdx(ii)=(f(ii+1)-f(ii-1))/(x(ii+1)-x(ii-1));
    end
end
end

        