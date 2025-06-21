function [Um,U_prime,U_2prime]=Load_DNSProfile(filename,y,Ret)
% load the DNS profiles

    data      = load(filename);

    if Ret ~=4179 && Ret ~=392.24 && Ret ~=10049
        ydelta    = [(data(:,1)-1);-flipud(data(1:end,1)-1)];
        Uplus     = [(data(:,3));flipud(data(1:end,3))];
    else
        ydelta    = [(data(:,1)-1);-flipud(data(1:end-1,1)-1)];
        Uplus     = [(data(:,3));flipud(data(1:end-1,3))];
    end



    % dUplusdy  = [(data(:,4));-flipud(data(:,4))]*Ret;
    dUplusdy  = diff2(Uplus,ydelta,size(ydelta,1))';
    d2Uplusdy = diff2_2nd(Uplus,ydelta,size(ydelta,1))';
    % linear interpolation onto Cheyshev grids
    Up     = interp1(ydelta,Uplus,    y,'linear','extrap');
    dUpdy  = interp1(ydelta,dUplusdy, y,'linear','extrap');
    d2Updy = interp1(ydelta,d2Uplusdy,y,'linear','extrap');
    Um         = diag(Up);
    U_prime    = diag(dUpdy);
    U_2prime   = diag(d2Updy);


end
