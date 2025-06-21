function d2fdx=diff2_2nd(f,x,nx)

    for ii=2:nx-1
        dxg(ii)=.5*(x(ii+1)-x(ii-1));
        fx(ii) =.5*(f(ii+1)-f(ii-1));
        d2xg(ii)=x(ii-1)-2*x(ii)+x(ii+1);
        fx2(ii) =f(ii-1)-2*f(ii)+f(ii+1);
    end
    dxg(1)=(-3*x(1)+4*x(2)-x(3))*0.5;
    dxg(nx)=(3*x(nx)-4*x(nx-1)+x(nx-2))*0.5;
    fx(1)=(-3*f(1)+4*f(2)-f(3))*0.5;
    fx(nx)=(3*f(nx)-4*f(nx-1)+f(nx-2))*0.5;
    
    d2xg(1) =x(1)-2*x(2)+x(3);  
    d2xg(nx)=x(nx-2)-2*x(nx-1)+x(nx);
    fx2(1)  =f(1)-2*f(2)+f(3);
    fx2(nx) =f(nx-2)-2*f(nx-1)+f(nx);

    for i=1:nx
        dcsidx (i) = 1.d0/(dxg(i));
        dcsidxs(i) = dcsidx(i)*dcsidx(i);
        dcsidx2(i) = -d2xg(i)*dcsidxs(i);

        dfdx(i) =fx(i)*dcsidx(i);
        d2fdx(i)=fx2(i)*dcsidxs(i)+dfdx(i)*dcsidx2(i);
    end

end