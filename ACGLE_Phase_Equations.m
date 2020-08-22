%% ETD-RK4 Algorithm to solve PDEs in the form
%  u(x,t)_t = L(u) + N(u)
%
% Computation domain is [-Lx,Lx]x[-Ly,Ly]. The wavenumbers should be 2pi/2L

%%% Choose PDE to solve %%%
% pdename: PDE to solve; %Choose: 'CGL',, 'FullPhase', 'Codim1', 'Codim1Beta', 'Codim2'. Defaults to CGL

%%% parameter_name: description (example) %%%
%%% See paper for specific examples
%
% delta: scaling parameter               (0.3)
% dt:    timestep size                   (0.01)
% tend:  end time                        (varies)
% N:     number of wavenumbers/grid size (512)
% Lx:    x-length. solve on [-Lx,Lx]     (150)
% Ly:    y-length. solve on [-Ly,Ly]     (150)
% alpha1: ACGLE parameter
% alpha2: ACGLE parameter  alpha1 == alpha2 -> CGLE
% beta:   ACGLE parameter
% kappa:  Phase-diffusion parameter   (-1)
% mu:     ACGLE bifurcation parameter (1)
% IC:     noise around zero or noise around one (0 or 1)
%         usually 0 for phase and 1 for ACGLE. 0 for ACGLE makes cells.


function sol = ETDRK4_PDE_Solver_GPU(pdename,delta,dt,tend,N,Lx,Ly,alpha1,alpha2,beta,kappa,mu,whichIC)

% Set plotting defaults
set(groot,'defaultAxesFontSize',25)
set(groot,'defaultLineMarkerSize',6)
set(groot,'defaultLineLineWidth',2)
screenInfo = get(0,'ScreenSize');
screenWidth = screenInfo(3);
screenHeight = screenInfo(4);
plotWidth = 1600;
plotHeight = 800;
willMovie = 1;                      % 0 or 1 for animation
rng(1);                             % For reproducability



fkeep = 1000; % Save every fkeep number of frames
switch pdename
    case 'CGL'
        tmax = tend;
    case 'FullPhase'
        tmax = tend;
    case 'Codim1'
        tmax = tend;
    case 'Codim1Beta'
        tmax = tend;
    case 'Codim2'
        tmax = tend;
    otherwise
        tmax = tend;
end

% Set parameters for Codim2 phase equation
r = 1; theta = 1.05*pi; kappa1 = r*cos(theta); kappa2 = r*sin(theta);

% This allows for scaling between ACGLE and the phase equations.
% For example in the codim2 equation, if Lx = 150 this function sets Lx <- delta*Lx.
[delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters(pdename, delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu);


%% Build current PDE
u0 = initial_conditions(pdename);
L = linear_operator(pdename); % Needed to build coeffs.
[E,E2,f1,f2,f3,Q,ind] = etdrk4_coeffs();

% Put data on GPU
u0 = gpuArray(u0);
L = gpuArray(L);
E = gpuArray(E);
E2 = gpuArray(E2);
f1 = gpuArray(f1);
f2 = gpuArray(f2);
f3 = gpuArray(f3);
Q = gpuArray(Q);
ind = gpuArray(ind);
v = fft2(u0);

t = 0;  % time
n = 0;  % loop iterations

if willMovie
    figure("Position",[screenWidth/3 screenHeight/5 plotWidth plotHeight]);
    vw = VideoWriter("ACGLE_movie.avi");
    open(vw);
end

% Main loop
while t <= tmax
    % main time stepping algorithm
    etdrk4_step(pdename);    

    % Animate
    if mod(n,fkeep) == 1 && willMovie
        plot_solution(ifft2(v),t);
        if willMovie
            frame = getframe(gcf);
            writeVideo(vw,frame)
        end
    end
    
    t = t + dt;
    n = n + 1;
end

if willMovie
    close(vw)
end

% Get solution at last computed time
sol = ifft2(v);

%% Wrappers
    function Lu = linear_operator(pdename)
        switch pdename
            case 'CGL'
                Lu = linear_operator_CGL();
            case 'FullPhase'
                Lu = linear_operator_phase();
            case 'Codim1'
                Lu = linear_operator_codim1();
            case 'Codim1Beta'
                Lu = linear_operator_codim1Beta();
            case 'Codim2'
                Lu = linear_operator_codim2();
            otherwise
                Lu = linear_operator_CGL();
        end
    end

    function etdrk4_step(pdename)
        switch pdename
            case 'CGL'
                etdrk4_step_CGL();
            case 'FullPhase'
                etdrk4_step_phase();
            case 'Codim1'
                etdrk4_step_codim1();
            case 'Codim1Beta'
                etdrk4_step_codim1Beta();
            case 'Codim2'
                etdrk4_step_codim2();
           otherwise
                etdrk4_step_CGL();
        end
    end

    function u0 = initial_conditions(pdename)
        switch pdename
            case 'CGL'
                u0 = initial_conditions_CGL();
            case 'FullPhase'
                u0 = initial_conditions_phase();
            case 'Codim1'
                u0 = initial_conditions_codim1();
            case 'Codim1Beta'
                u0 = initial_conditions_codim1Beta();
            case 'Codim2'
                u0 = initial_conditions_codim2();
            otherwise
                u0 = initial_conditions_CGL();
        end
    end

    function [delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters(pdename,delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu)
        switch pdename
            case 'CGL'
                [delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters_CGL(delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu);
            case 'FullPhase'
                [delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters_phase(delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu);
            case 'Codim1'
                [delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters_codim1(delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu);
            case 'Codim1Beta'
                [delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters_codim1Beta(delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu);
            case 'Codim2'
                [delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters_codim2(delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu);               
            otherwise
                [delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters_CGL(delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu);
        end
    end

%% Start of PDE specific functions.
%% Codim 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L(u). The linear part of the PDE
    function Lu = linear_operator_codim1()
        [Kx, Ky] = wavenumbers_codim1();
        L2 = kappa*Kx.^2 + (1+alpha2*beta)*Ky.^2;
        L4 = -(1+1/beta^2)/(2*mu)*Kx.^4;
        Lu = -L2 + L4;
    end

% N(u). The nonlinear part of the PDE. Takes function in spatial domain.
% Returns function in Fourier domain.
    function Nu = nonlinear_operator_codim1(f)
        [fx,~] = periodic_gradient(f, 2*Lx/(N-1), 2*Ly/(N-1));
        Nu = (beta+1/beta)*fx.^2;
    end

% Make wavenumbers
    function [Kx, Ky] = wavenumbers_codim1()
        kx = 2*pi*[0:N/2-1 0 -N/2+1:-1]'/(2*Lx); % 2pi/2L since range from -L to L
        ky = 2*pi*[0:N/2-1 0 -N/2+1:-1]'/(2*Ly);
        [Kx, Ky] = meshgrid(kx,ky);
    end

% Update the solution according the etdrk4.
    function etdrk4_step_codim1()
        Nv = fft2(nonlinear_operator_codim1(ifft2(v))); %Nonlinear evaluation. g(u,*)
        a = E2.*v + Q.*Nv; %Coefficient ’a’ in ETDRK formula
        Na = fft2(nonlinear_operator_codim1(ifft2(a))); %Nonlinear evaluation. g(a,*)
        b = E2.*v + Q.*Na; %Coefficient ’b’ in ETDRK formula
        Nb = fft2(nonlinear_operator_codim1(ifft2(b))); %Nonlinear evaluation. g(b,*)
        c = E2.*a + Q.*(2*Nb-Nv); %Coefficient ’c’ in ETDRK formula
        Nc = fft2(nonlinear_operator_codim1(ifft2(c))); %Nonlinear evaluation. g(c,*)
        v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; %update
        v(ind) = 0;
        u = ifft2(v);
        u = u - mean(mean(u));
        v = fft2(u);
    end

% Set up the initial conditions
    function ic = initial_conditions_codim1()
        %[X,Y] = makegrid(); Can use this if you want to make your own IC
        if whichIC == 0
            ic = .01*rand(N,N);
        elseif whichIC == 1 
            ic = ones(N,N) + .01*rand(N,N);
        else
            error("Bad IC choice. Pick 0 or 1")
        end
        
        ic = ic - mean(mean(ic));
    end

% Set problem specific parameters
    function [delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters_codim1(delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu)
        Lx = delta*Lx;
        Ly = delta^2*Ly;
    end

%% Codim 1 with Beta dependence %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L(u). The linear part of the PDE
    function Lu = linear_operator_codim1Beta()
        [Kx, Ky] = wavenumbers_codim1Beta();
        L2 = Kx.^2 + Ky.^2;
        L4 = -(1+1/beta^2)/(2*mu)*Kx.^4;
        Lu = -L2 + L4;
    end

% N(u). The nonlinear part of the PDE. Takes function in spatial domain.
% Returns function in Fourier domain.
    function Nu = nonlinear_operator_codim1Beta(f)
        [fx,~] = gradient(f, 2*Lx/(N-1), 2*Ly/(N-1));
        Nu = -(beta+1/beta)*fx.^2;
    end

% Make wavenumbers
    function [Kx, Ky] = wavenumbers_codim1Beta()
        kx = 2*pi*[0:N/2-1 0 -N/2+1:-1]'/(2*Lx); % Maybe times delta
        ky = 2*pi*[0:N/2-1 0 -N/2+1:-1]'/(2*Ly);
        [Kx, Ky] = meshgrid(kx,ky);
    end

% Update the solution according the etdrk4.
    function etdrk4_step_codim1Beta()
        Nv = fft2(nonlinear_operator_codim1Beta(ifft2(v))); %Nonlinear evaluation. g(u,*)
        a = E2.*v + Q.*Nv; %Coefficient ’a’ in ETDRK formula
        Na = fft2(nonlinear_operator_codim1Beta(ifft2(a))); %Nonlinear evaluation. g(a,*)
        b = E2.*v + Q.*Na; %Coefficient ’b’ in ETDRK formula
        Nb = fft2(nonlinear_operator_codim1Beta(ifft2(b))); %Nonlinear evaluation. g(b,*)
        c = E2.*a + Q.*(2*Nb-Nv); %Coefficient ’c’ in ETDRK formula
        Nc = fft2(nonlinear_operator_codim1Beta(ifft2(c))); %Nonlinear evaluation. g(c,*)
        v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; %update
        v(ind) = 0;
        u = ifft2(v);
        u = u - mean(mean(u));
        v = fft2(u);
    end

% Set up the initial conditions
    function ic = initial_conditions_codim1Beta()
        %[X,Y] = makegrid(); Can use this if you want to make your own IC
        if whichIC == 0
            ic = .01*rand(N,N);
        elseif whichIC == 1 
            ic = ones(N,N) + .01*rand(N,N);
        else
            error("Bad IC choice. Pick 0 or 1")
        end
        ic = ic - mean(mean(ic));
    end

% Set problem specific parameters
    function [delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters_codim1Beta(delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu)
        Lx = delta*Lx;
        Ly = delta^2*Ly;
    end

%% Codim 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L(u). The linear part of the PDE
    function Lu = linear_operator_codim2()
        [Kx, Ky] = wavenumbers_codim2();
        L2 = kappa1*Kx.^2 + kappa2*Ky.^2;
        L4 = -(1+1/beta^2)/(2*mu)*(Kx.^4 + 2*Kx.^2.*Ky.^2 + Ky.^4);
        Lu = -L2 + L4;
    end

% N(u). The nonlinear part of the PDE. Takes function in spatial domain.
% Returns function in Fourier domain.
    function Nu = nonlinear_operator_codim2(f)
        [fx,fy] = periodic_gradient(f, 2*Lx/(N-1), 2*Ly/(N-1));
        Nu = (beta+1/beta)*(fx.^2 + fy.^2);
    end

% Make wavenumbers
    function [Kx, Ky] = wavenumbers_codim2()
        kx = 2*pi*[0:N/2-1 0 -N/2+1:-1]'/(2*Lx); % Maybe times delta
        ky = 2*pi*[0:N/2-1 0 -N/2+1:-1]'/(2*Ly); % Maybe times delta
        [Kx, Ky] = meshgrid(kx,ky);
    end

% Update the solution according the etdrk4.
    function etdrk4_step_codim2()
        Nv = fft2(nonlinear_operator_codim2(ifft2(v))); %Nonlinear evaluation. g(u,*)
        a = E2.*v + Q.*Nv; %Coefficient ’a’ in ETDRK formula
        Na = fft2(nonlinear_operator_codim2(ifft2(a))); %Nonlinear evaluation. g(a,*)
        b = E2.*v + Q.*Na; %Coefficient ’b’ in ETDRK formula
        Nb = fft2(nonlinear_operator_codim2(ifft2(b))); %Nonlinear evaluation. g(b,*)
        c = E2.*a + Q.*(2*Nb-Nv); %Coefficient ’c’ in ETDRK formula
        Nc = fft2(nonlinear_operator_codim2(ifft2(c))); %Nonlinear evaluation. g(c,*)
        v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; %update
        v(ind) = 0;
        u = ifft2(v);
        u = u - mean(mean(u));
        v = fft2(u);
    end

% Set up the initial conditions
    function ic = initial_conditions_codim2()
        %[X,Y] = makegrid(); Can use this if you want to make your own IC
        if whichIC == 0
            ic = .01*rand(N,N);
        elseif whichIC == 1 
            ic = ones(N,N) + .01*rand(N,N);
        else
            error("Bad IC choice. Pick 0 or 1")
        end
        ic = ic - mean(mean(ic));
    end

% Set problem specific parameters
    function [delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters_codim2(delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu)
        Lx = delta*Lx;
        Ly = delta*Ly;
    end


%% Full Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L(u). The linear part of the PDE
    function Lu = linear_operator_phase()
        [Kx, Ky] = wavenumbers_phase();
        L2 = ((1+alpha1*beta)*Kx.^2 + (1+alpha2*beta)*Ky.^2)/delta^2;
        L4 = ((beta*alpha1-alpha1^2)*Kx.^4 + (beta*(alpha1+alpha2)-2*alpha1*alpha2)*Kx.^2.*Ky.^2 + (beta*alpha2-alpha2^2)*Ky.^4)/(2*mu);
        Lu = -L2 + L4;
    end

% N(u). The nonlinear part of the PDE. Takes function in spatial domain.
% Returns function in Fourier domain.
    function Nu = nonlinear_operator_phase(f)
        [fx,fy] = periodic_gradient(f, 2*Lx/(N-1), 2*Ly/(N-1));
        Nu = (beta-alpha1)*fx.^2 + (beta-alpha2)*fy.^2;
    end

% Make wavenumbers
    function [Kx, Ky] = wavenumbers_phase()
        kx = 2*pi*[0:N/2-1 0 -N/2+1:-1]'/(2*Lx); % 2pi/2L since range from -L to L
        ky = 2*pi*[0:N/2-1 0 -N/2+1:-1]'/(2*Ly);
        [Kx, Ky] = meshgrid(kx,ky);
    end

% Update the solution according the etdrk4.
    function etdrk4_step_phase()
        Nv = fft2(nonlinear_operator_phase(ifft2(v))); %Nonlinear evaluation. g(u,*)
        a = E2.*v + Q.*Nv; %Coefficient ’a’ in ETDRK formula
        Na = fft2(nonlinear_operator_phase(ifft2(a))); %Nonlinear evaluation. g(a,*)
        b = E2.*v + Q.*Na; %Coefficient ’b’ in ETDRK formula
        Nb = fft2(nonlinear_operator_phase(ifft2(b))); %Nonlinear evaluation. g(b,*)
        c = E2.*a + Q.*(2*Nb-Nv); %Coefficient ’c’ in ETDRK formula
        Nc = fft2(nonlinear_operator_phase(ifft2(c))); %Nonlinear evaluation. g(c,*)
        v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; %update
        v(ind) = 0;
        u = ifft2(v);
        u = u - mean(mean(u)); % has a growing mean so need to subtract it away
        v = fft2(u);
    end

% Set up the initial conditions
    function ic = initial_conditions_phase()
        %[X,Y] = makegrid(); Can use this if you want to make your own IC
        if whichIC == 0
            ic = .01*rand(N,N);
        elseif whichIC == 1 
            ic = ones(N,N) + .01*rand(N,N);
        else
            error("Bad IC choice. Pick 0 or 1")
        end
        ic = ic - mean(mean(ic));
    end

% Set problem specific parameters
    function [delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters_phase(delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu)
        Lx = delta*Lx;
        Ly = delta*Ly;
    end


%% CGL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L(u). The linear part of the PDE
    function Lu = linear_operator_CGL()
        [Kx, Ky] = wavenumbers_CGL();
        L2 = (1+1i*alpha1).*Kx.^2 + (1+1i*alpha2).*Ky.^2;
        Lu = -L2;
    end

% N(u). The nonlinear part of the PDE. Takes function in spatial domain.
% Returns function in Fourier domain.
    function Nu = nonlinear_operator_CGL(f)
        Nu = mu.*f - (1+1i*beta).*abs(f).^2.*f;
        %         Nu = mu.*f - (beta-1i).*abs(f).^2.*f;
    end

% Make wavenumbers
    function [Kx, Ky] = wavenumbers_CGL()
        kx = 2*pi*[0:N/2-1 0 -N/2+1:-1]'/(2*Lx);
        ky = 2*pi*[0:N/2-1 0 -N/2+1:-1]'/(2*Ly);
        [Kx, Ky] = meshgrid(kx,ky);
    end

% Update the solution according the etdrk4.
    function etdrk4_step_CGL()
        
        Nv = fft2(nonlinear_operator_CGL(ifft2(v))); %Nonlinear evaluation. g(u,*)
        a = E2.*v + Q.*Nv; %Coefficient ’a’ in ETDRK formula
        Na = fft2(nonlinear_operator_CGL(ifft2(a))); %Nonlinear evaluation. g(a,*)
        b = E2.*v + Q.*Na; %Coefficient ’b’ in ETDRK formula
        Nb = fft2(nonlinear_operator_CGL(ifft2(b))); %Nonlinear evaluation. g(b,*)
        c = E2.*a + Q.*(2*Nb-Nv); %Coefficient ’c’ in ETDRK formula
        Nc = fft2(nonlinear_operator_CGL(ifft2(c))); %Nonlinear evaluation. g(c,*)
        v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; %update
        v(ind) = 0;
    end

% Set up the initial conditions
    function ic = initial_conditions_CGL()
        %[X,Y] = makegrid();
        if whichIC == 0
            ic = .01*randn(N,N); % randn!
        elseif whichIC == 1 
            ic = ones(N,N) + .01*rand(N,N);
        else
            error("Bad IC choice. Pick 0 or 1")
        end
    end

% Set problem specific parameters
    function [delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu] = set_parameters_CGL(delta, Lx, Ly, alpha1, alpha2, beta, kappa, mu)
%         alpha1 = (kappa*delta^2-1)/beta; 
    end

%% END PDEs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Generic/Helper Function
% Plot the solution
    function plot_solution(u,time)
        %[X,Y] = makegrid();
        subplot(1,2,1)
        pcolor(abs(u)); 
        shading interp; title(time); axis image; colorbar; colormap(parula);
        xlabel("abs")
        set(gca,'XTickLabel',[],'YTickLabel',[]);
        
        subplot(1,2,2)
        pcolor(real(u));
        shading interp; axis image; colorbar; colormap(parula);
        xlabel("real part")
        set(gca,'XTickLabel',[],'YTickLabel',[]);
        drawnow;
    end

% Build spatial grid
    function [X,Y] = makegrid()
        x = -Lx:2*Lx/(N-1):Lx;
        y = -Ly:2*Ly/(N-1):Ly;
        [X,Y] = meshgrid(x',y');
    end

% Build the coefficients
    function [E,E2,f1,f2,f3,Q,ind] = etdrk4_coeffs()
        E = exp(dt*L);
        E2 = exp(dt*L/2);
        M = 16; % no. of points for complex mean
        r = exp(1i*pi*((1:M)-0.5)/M); % roots of unity
        L = L(:); LR = dt*L(:,ones(M,1)) + r(ones(N^2,1),:); % Change power of N for dimension.
        Q = dt*real(mean((exp(LR/2)-1)./LR, 2));
        f1 = dt*real(mean((-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3, 2));
        f2 = dt*real(mean((2+LR+exp(LR).*(-2+LR))./LR.^3, 2));
        f3 = dt*real(mean((-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3, 2));
        f1 = reshape(f1,N,N); f2 = reshape(f2,N,N); f3 = reshape(f3,N,N);
        Q = reshape(Q,N,N);
        Fr = false(N,1); %High frequencies for de-aliasing
        Fr(N/2+1-round(N/9) : N/2+round(N/9)) = 1;
        [alxi,aleta] = meshgrid(Fr,Fr); %{[alxi,aleta,alzeta]=meshgrid(Fr,Fr,Fr)}
        ind = alxi | aleta; %de-aliasing index. {alxi | aleta | alzeta}
    end

% Periodic Gradient
    function [fx,fy] = periodic_gradient(f,varargin)
        if nargin == 1
            f_pad = gpuArray(zeros(size(f)+2));
            f_pad(2:end-1,2:end-1) = f;
            f_pad(1,2:end-1) = f(end,:);
            f_pad(end,2:end-1) = f(1,:);
            f_pad(2:end-1,1) = f(:,end);
            f_pad(2:end-1,end) = f(:,1);
            f_pad(1,1) = f(end,end);
            f_pad(end,end) = f(1,1);
            f_pad(1,end) = f(end,1);
            f_pad(end,1) = f(1,end);
            [fx_pad,fy_pad] = gradient(f_pad);
            fx = fx_pad(2:end-1,2:end-1);
            fy = fy_pad(2:end-1,2:end-1);
        elseif nargin == 3
            xs = varargin(1);
            ys = varargin(2);
            f_pad = gpuArray(zeros(size(f)+2));
            f_pad(2:end-1,2:end-1) = f;
            f_pad(1,2:end-1) = f(end,:);
            f_pad(end,2:end-1) = f(1,:);
            f_pad(2:end-1,1) = f(:,end);
            f_pad(2:end-1,end) = f(:,1);
            f_pad(1,1) = f(end,end);
            f_pad(end,end) = f(1,1);
            f_pad(1,end) = f(end,1);
            f_pad(end,1) = f(1,end);
            [fx_pad,fy_pad] = gradient(f_pad,cell2mat(xs),cell2mat(ys));
            fx = fx_pad(2:end-1,2:end-1);
            fy = fy_pad(2:end-1,2:end-1);
        end
        
    end

end