close all
clear all

% resolvent estimation based on stochastic forcing (with/without viscosity model)
% for incompressible channel flows
% by fanyitong 30/5/2023 == A TEST (1st) ==
%% the parameters of the case
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for casenum = 1:1
    switch casenum
        case(1)
        filename='chan_Re180.txt';
        Ret = 182.088 ; u_tau = 6.37309e-02; % test on chan_Re180
        routename = 'Re180';
        case(2)
        filename='chan_Re550.txt';
        Ret = 543.496 ; u_tau = 5.43496e-02;
        routename = 'Re550';
        case(3)
        filename='chan_Re1000.txt';
        Ret = 1000.512 ; u_tau = 5.00256e-02;
        routename = 'Re1000';
        case(4)
        filename='chan_Re2000.txt';
        Ret = 1994.756 ; u_tau = 4.58794e-02;
        routename = 'Re2000';
        case(5)
        filename='chan_Re5200.txt';
        Ret = 5185.897  ; u_tau = 4.14872e-02 ;
        routename = 'Re5200';
        case(6)
        filename='Re4200.prof';
        Ret = 4179 ; u_tau =0.042510839221638;
        routename = 'Re4200'; 
        case(7)
        filename='chan_Re380.txt';
        Ret = 392.24  ; u_tau = 0.056997103681638;
        routename = 'Re380';
        case(8)
        filename='chan_Re10000.txt';
        Ret = 10049  ; u_tau =  0.034629;
        routename = 'Re10000';

    end
    iflag = 0; % 0 and 1 for with/without eddy viscosity model
    
    % the dataset of DNS nx*ny*nz = 1024*192*512; Lx*Ly*Lz = 8pi*2*3pi
    % the wavenumbers and frequencies
    %lambdax = 700/Ret; lambdaz = 100/Ret; cPs = 15; %(Towne2020)
    %kxs = 2*pi/lambdax; % streamwise wavenumber
    %kzs = 2*pi/lambdaz; % spanwise wavenumber
    % kx=1;kz=2.67;cp=20; %(Symon2020)
    % omega = kx* cp; % frequency
    
    nsvd = 2; % number of singular values to compute
    N =400 ; % point numbers in the wall-normal direction 
    %y=cos([0:N-1]'*pi/(N-1));
    
    
    
    % % the wavenumbers and frequencies
    Lamx = [log(0.01):0.1:4];
    Lamz = [log(0.01):0.05:4];
    lambdaxs = exp(Lamx);
    lambdazs = exp(Lamz);
    kxs   = 2*pi./lambdaxs; % streamwise wavenumber series
    kzs   = 2*pi./lambdazs; % spanwise wavenumber series
    cPs  = [0.05:0.025:1.2]/u_tau; % wavespeed series (corresponding to the critical layer)
    
    [y,DM] = chebdif(N,2); 
    
    [Um,U_prime,U_2prime]=Load_DNSProfile(filename,y,Ret); % load DNS-PROFILE
    
    
    
    SS = zeros(nsvd,length(kxs),length(kzs),length(cPs));
    TotalE = zeros(length(kxs),length(kzs),length(cPs));
    SU = zeros(4*N,nsvd,length(kxs),length(kzs),length(cPs));
    SV = zeros(3*N,nsvd,length(kxs),length(kzs),length(cPs));
    
    for cc = 1:length(cPs)
        for iz = 1:1:length(kzs)
            for ix = 1:1:length(kxs)
                tic
                cP = cPs(cc); kz = kzs(iz); kx = kxs(ix);
                [u0, s0, v0, SumSvalSq,s0all] = EstablishResolvent(Ret,kx,kz,cP,N,nsvd,Um,U_prime,U_2prime);
                SS(1:nsvd,ix,iz,cc) = s0;
                TotalE(ix,iz,cc) = SumSvalSq;
                SU(:,:,ix,iz,cc) = u0;
                SV(:,:,ix,iz,cc) = v0;
    
                disp([num2str(ix) ', ' num2str(iz) ', ' num2str(cc) ' (' num2str(length(kxs)) ', ' num2str(length(kzs)) ', ' num2str(length(cPs)) ')'])
                disp([num2str(cc) ': spectral energy : ' num2str(sum(SS(:,cc).^2)/SumSvalSq)])
                toc
            end
        end
    end
    
    UU = SU(1:N,:,:,:,:);
    VV = SU(N+1:2*N,:,:,:,:);
    WW = SU(2*N+1:3*N,:,:,:,:);
    PP = SU(3*N+1:4*N,:,:,:,:);
    save(['u_response_mode.mat'],'UU','-v7.3');
    save(['v_response_mode.mat'],'VV','-v7.3');
    save(['w_response_mode.mat'],'WW','-v7.3');
    save(['p_response_mode.mat'],'PP','-v7.3');
    FUU = SV(1:N,:,:,:,:);
    FVV = SV(N+1:2*N,:,:,:,:);
    FWW = SV(2*N+1:3*N,:,:,:,:);
    save(['u_forcing_mode.mat'],'FUU','-v7.3');
    save(['v_forcing_mode.mat'],'FVV','-v7.3');
    save(['w_forcing_mode.mat'],'FWW','-v7.3');
    clear SU SV UU VV WW PP FUU FVV FWW
    save(['forsigma_3d.mat'],'-v7.3');

    clearvars -except casenum

end



return

%%
figure()
ii = 1;plot(y(1:end),abs(SU((ii-1)*N+1:ii*N,1)),'linewidth',2);hold on
ii = 1;plot(y(1:end),abs(SU((ii-1)*N+1:ii*N,2)),'linewidth',2);hold on
xlim([-1 1]);
figure()
ii=2;plot(y(1:end),abs(SU((ii-1)*N+1:ii*N,1)),'linewidth',2);hold on
ii=2;plot(y(1:end),abs(SU((ii-1)*N+1:ii*N,2)),'linewidth',2);hold on
xlim([-1 1]);
figure()
ii=3;plot(y(1:end),abs(SU((ii-1)*N+1:ii*N,1)),'linewidth',2);hold on
ii=3;plot(y(1:end),abs(SU((ii-1)*N+1:ii*N,2)),'linewidth',2);hold on
xlim([-1 1]);
figure()
ii = 4;plot(y(1:end),abs(SU((ii-1)*N+1:ii*N,1)),'linewidth',2);hold on
ii = 4;plot(y(1:end),abs(SU((ii-1)*N+1:ii*N,2)),'linewidth',2);hold on
xlim([-1 1]);

figure()
plot([1:100],s0all(1:100),'o');hold on
set(gca,'yscale','log')


return
%%
close all
clear all
load forsigma_3d.mat
figure()
pcolor(lambdaxs*Ret,lambdazs*Ret,((squeeze(SS.^2))./TotalE)');shading interp
set(gca,'xscale','log','yscale','log')
caxis([0 1])



