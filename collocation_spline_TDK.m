function [T,dpdc,lambda,coeffs,res,exitflag,dedxi,dedq,dpdL,dnbdc,dnbbdL] = collocation_spline_TDK(qa,fe,Fe,Me,init_guess,GV)
% Implementation based on cubic B-spline
% https://www.math.ucdavis.edu/~bremer/classes/fall2018/MAT128a/lecture15.pdf
% Mar 2024, Yifan Wang
% Kirchhoff tendon-driven robot
% 

% tau = zeros(GV.num_tendons,1);
% if qa(1) >= 0, tau(1) = qa(1); else, tau(3) = -qa(1); end
% if qa(2) >= 0, tau(2) = qa(2); else, tau(4) = -qa(2); end
tau = [qa(1:2);-qa(1:2)]; % push-pull
% hsp0 = GV.hsp; D0 = GV.D;
GV.hsp = GV.hsp*qa(3); % 1xn
GV.D = GV.D/qa(3);
GV.dUdotdc = GV.dUdotdc/qa(3);
%     counter = 0;
global T dT Tq dTq lambda;
T = zeros(4,4,GV.n+1); T(:,:,1) = eye(4);
dT = zeros(4,4,3*(GV.dof),GV.n+1);
Tq = zeros(4,4,GV.n*3);
dTq = zeros(4,4,3*(GV.dof),GV.n*3); % 4x4x3dofxvn
% [res,dedxi] = collocation_error(init_guess); exitflag = true; xic = init_guess;

[coeffs,res,exitflag,output,dedxi] = fsolve(@(coeffs) collocation_sp_error(coeffs), init_guess, GV.options_sp);

% coeffs = reshape(coeffs,[3 GV.dof]);

[dedtau,dedL,dTdL_,dnbdc,dnbbdL] = collocation_Jacobian(coeffs,qa(3)); dedtau = dedtau*[1 0;0 1;-1 0;0 -1];
dedq = [dedtau,dedL];
dpdc = squeeze(dT(1:3,4,:,end));
dpdL = dTdL_(1:3,4,end);

% eta_hat = pagemrdivide(tensorprod(dT(:,:,:,end),(dedxi\dedtau),3,1),T(:,:,end)); % 4x4xnum_tendons
% Jacobian = squeeze([eta_hat(3,2,:);eta_hat(1,3,:);eta_hat(2,1,:);eta_hat(1,4,:);eta_hat(2,4,:);eta_hat(3,4,:)]);
% Jacobian(4:6,:) = Jacobian(4:6,:) + cross(Jacobian(1:3,:), repmat(T(1:3,4,end),1,GV.num_tendons));

    function [e,J] = collocation_sp_error(coeffs)
        % coeffs: 3dofx1 spline coefficients stacked
        % Coeffs: 3xdof spline coefficients
        % calculate pose at collocation points via magnus expansion
        Coeffs = reshape(coeffs,[3 GV.dof]);
        uq = GV.Q*Coeffs'; % vn x 3, strain at quadrature points
        Uc = Coeffs*GV.S'; % 3x(n+1), strain at collocation points
        dPsidB = zeros(6,6,3);
        D_hat = zeros(4,4,3*GV.dof);
        Dq1_hat = zeros(4,4,3*GV.dof);
        Dq2_hat = zeros(4,4,3*GV.dof);
        Dq3_hat = zeros(4,4,3*GV.dof);

        for k = 1:GV.n % forward kinematics propagation, calculate SE3 transformation and derivatives
            % snapshots at quadrature point
            Xq = [uq(k*3-2:k*3,:)  repmat([0 0 1],3,1)]; % vx6

            % Calculate Magnus expansion
            B = (GV.invV*Xq*GV.hsp(k))'; % 6xv self-adjoint basis
            B1 = B(:,1); B2 = B(:,2); B3 = B(:,3);
            w1_hat = [0 -B1(3) B1(2);
                B1(3) 0 -B1(1);
                -B1(2) B1(1) 0];
            v1_hat = [0 -B1(6) B1(5);
                B1(6) 0 -B1(4);
                -B1(5) B1(4) 0];
            adB1 = [w1_hat zeros(3);v1_hat w1_hat];
            w2_hat = [0 -B2(3) B2(2);
                B2(3) 0 -B2(1);
                -B2(2) B2(1) 0];
            v2_hat = [0 -B2(6) B2(5);
                B2(6) 0 -B2(4);
                -B2(5) B2(4) 0];
            adB2 = [w2_hat zeros(3);v2_hat w2_hat];
            w3_hat = [0 -B3(3) B3(2);
                B3(3) 0 -B3(1);
                -B3(2) B3(1) 0];
            v3_hat = [0 -B3(6) B3(5);
                B3(6) 0 -B3(4);
                -B3(5) B3(4) 0];
            adB3 = [w3_hat zeros(3);v3_hat w3_hat];

            B12 = adB1*B2;
            B23 = adB2*B3;
            B13 = adB1*B3;
            B113 = adB1*B13;
            B212 = adB2*B12;
            B112 = adB1*B12;
            B1112 = adB1*B112;

            Psi = B1 + B3/12 + B12/12 - B23/240 + B113/360 - B212/240 - B1112/720;

            % Blane's method
            %             B13 = B1/12 + B3/240;
            %             adB13 = adB1/12 + adB3/240;
            %
            %             %C2 = B13*B2 - B2*B13;
            %             C2 = adB13*B2;
            %             %B2C2 = B2*C2 - C2*B2;
            % %             wC2_hat = [0 -C2(3) C2(2);
            % %                 C2(3) 0 -C2(1);
            % %                 -C2(2) C2(1) 0];
            % %             vC2_hat = [0 -C2(6) C2(5);
            % %                 C2(6) 0 -C2(4);
            % %                 -C2(5) C2(4) 0];
            %             %adC2 = [wC2_hat zeros(3);vC2_hat wC2_hat];
            %             B2C2 = adB2*C2;
            %             B3C2 = B3/360 - C2/60;
            %             %adB3C2 = adB3/360 - adC2/60;
            %
            %             %C132 = B1*B3C2 - B3C2*B1;
            %             C132 = adB1*B3C2;
            %             %C3 = B1*C132 - C132*B1;
            %             C3 = adB1*C132;
            %             Psi = B1 + B3/12 + C2 - B2C2/20 + C3;

            % exponential map se3 to SE3
            [expPsi,dexp_Psi] = expSE3(Psi);

            % Integrate on SE3
            T(:,:,k+1) = T(:,:,k)*expPsi;

            % assume constant curvature between quadrature points
            Psiq1 = (Xq(1,:)' + [Uc(:,k);0;0;1])/2*GV.hsp(k)*(1/2-sqrt(15)/10);
            Psiq2 = (Xq(2,:)' + Xq(1,:)')/2*GV.hsp(k)*sqrt(15)/10;
            Psiq3 = (Xq(3,:)' + [Uc(:,k+1);0;0;1])/2*GV.hsp(k)*(1/2-sqrt(15)/10);
            [expPsiq1,dexp_Psiq1] = expSE3(Psiq1);
            [expPsiq2,dexp_Psiq2] = expSE3(Psiq2);
            [expPsiq3,dexp_Psiq3] = expSE3(Psiq3);
            Tq(:,:,k*3-2) = T(:,:,k)*expPsiq1;
            Tq(:,:,k*3-1) = Tq(:,:,k*3-2)*expPsiq2;
            Tq(:,:,k*3) = T(:,:,k+1)/expPsiq3;


            % Derivatives on SE3
            % from uc to uq
            % duqdc = zeros(3,3*GV.dof,3);
            duqdc = GV.dUqdc(:,:,k); % 3vx3dof

            % from uq to B
            dXdu = GV.hsp(k)*[eye(3);zeros(3)]; % 6x3

            % from A to B
            %dBdA = GV.invV; % vxv

            % from B to Psi
            w12_hat = [0 -B12(3) B12(2);
                B12(3) 0 -B12(1);
                -B12(2) B12(1) 0];
            v12_hat = [0 -B12(6) B12(5);
                B12(6) 0 -B12(4);
                -B12(5) B12(4) 0];
            adB12 = [w12_hat zeros(3);v12_hat w12_hat];
            w13_hat = [0 -B13(3) B13(2);
                B13(3) 0 -B13(1);
                -B13(2) B13(1) 0];
            v13_hat = [0 -B13(6) B13(5);
                B13(6) 0 -B13(4);
                -B13(5) B13(4) 0];
            adB13 = [w13_hat zeros(3);v13_hat w13_hat];
            w112_hat = [0 -B112(3) B112(2);
                B112(3) 0 -B112(1);
                -B112(2) B112(1) 0];
            v112_hat = [0 -B112(6) B112(5);
                B112(6) 0 -B112(4);
                -B112(5) B112(4) 0];
            adB112 = [w112_hat zeros(3);v112_hat w112_hat];
            dPsidB(:,:,1) = eye(6) - adB2/12 - (adB1*adB3 + adB13)/360 +...
                adB2*adB2/240 + (adB1*adB1*adB2 + adB1*adB12 + adB112)/720; % 6x6
            dPsidB(:,:,2) = adB1/12 + adB3/240 + (-adB2*adB1 + adB12)/240 -...
                adB1*adB1*adB1/720; % 6x6
            dPsidB(:,:,3) = eye(6)/12 - adB2/240 + adB1*adB1/360; % 6x6

            % from Psi to dR
            dPsidc = zeros(6,3*GV.dof);
            for i = 1:3
                for j = 1:3
                    % derivative through Bi to Aj (uqj)
                    dPsidc = dPsidc + dPsidB(:,:,i)*GV.invV(i,j)*dXdu*duqdc(j*3-2:j*3,:);
                end
            end
            D = dexp_Psi*dPsidc; % 6x3dof
            for i = 1:3*GV.dof % can be reduced
                D_hat(:,:,i) = [0 -D(3,i) D(2,i) D(4,i);
                    D(3,i) 0 -D(1,i) D(5,i);
                    -D(2,i) D(1,i) 0 D(6,i);
                    0 0 0 0];
            end

            dT(:,:,:,k+1) = pagemtimes(dT(:,:,:,k) + pagemtimes(T(:,:,k),D_hat),expPsi); % 4x4x3dof
            
            dPsiq1dc = dXdu*(duqdc(1:3,:) + GV.dUcdc(:,:,k))/2*(1/2-sqrt(15)/10); % 6x3dof
            Dq1 = dexp_Psiq1*dPsiq1dc; % 6x3dof
            for i = 1:3*GV.dof
                Dq1_hat(:,:,i) = [0 -Dq1(3,i) Dq1(2,i) Dq1(4,i);
                    Dq1(3,i) 0 -Dq1(1,i) Dq1(5,i);
                    -Dq1(2,i) Dq1(1,i) 0 Dq1(6,i);
                    0 0 0 0];
            end
            dTq(:,:,:,k*3-2) = pagemtimes(dT(:,:,:,k) + pagemtimes(T(:,:,k),Dq1_hat),expPsiq1);
            dPsiq2dc = dXdu*(duqdc(1:3,:) + duqdc(4:6,:))/2*sqrt(15)/10; % 6x3dof
            Dq2 = dexp_Psiq2*dPsiq2dc; % 6x3dof
            for i = 1:3*GV.dof
                Dq2_hat(:,:,i) = [0 -Dq2(3,i) Dq2(2,i) Dq2(4,i);
                    Dq2(3,i) 0 -Dq2(1,i) Dq2(5,i);
                    -Dq2(2,i) Dq2(1,i) 0 Dq2(6,i);
                    0 0 0 0];
            end
            dTq(:,:,:,k*3-1) = pagemtimes(dTq(:,:,:,k*3-2) + pagemtimes(Tq(:,:,k*3-2),Dq2_hat),expPsiq2);
            dPsiq3dc = dXdu*(duqdc(7:9,:) + GV.dUcdc(:,:,k+1))/2*(1/2-sqrt(15)/10); % 6x3dof
            Dq3 = dexp_Psiq3*dPsiq3dc; % 6x3dof
            for i = 1:3*GV.dof
                Dq3_hat(:,:,i) = [0 -Dq3(3,i) Dq3(2,i) Dq3(4,i);
                    Dq3(3,i) 0 -Dq3(1,i) Dq3(5,i);
                    -Dq3(2,i) Dq3(1,i) 0 Dq3(6,i);
                    0 0 0 0];
            end
            dTq(:,:,:,k*3) = pagemrdivide(dT(:,:,:,k+1),expPsiq3) - pagemtimes(Tq(:,:,k*3),Dq3_hat);
        end

        % Setup tendon linear system
        gx = zeros(3,GV.n+1);
        dg = zeros(3*(GV.n+1),3*GV.dof);
        % nbt = zeros(3,GV.n+1); % internal actuation force
        % dgLu = squeeze(pagemtimes(dT(1:3,1:3,:,end),'transpose',Me,'none'));
        dgLu = zeros(3);
        %intfe = zeros(3,GV.n+1);
        % quadrature for integration of distributed external force
        [fei,dfei] = fe(squeeze(Tq(1:3,4,:))); dfei = squeeze(dfei); % 3x3xvn
        intfe = [GV.hsp.*(5*(fei(:,1:3:end) + fei(:,3:3:end)) + 8*fei(:,2:3:end))/18 Fe];
        Intfe = cumsum(intfe,2,"reverse");
        % dpq = squeeze(dTq(1:3,4,:,:)); % 3x3dofxvn
        dfedc = pagemtimes(dfei,squeeze(dTq(1:3,4,:,:))); % 3x3dofxvn
        dintfedc = cat(3,...
            reshape(GV.hsp,1,1,[]).*(5*(dfedc(:,:,1:3:end) + dfedc(:,:,3:3:end)) + 8*dfedc(:,:,2:3:end))/18,...
            zeros(3,3*GV.dof)); % 3x3dofx(n+1)
        dIntfedc = cumsum(dintfedc,3,"reverse"); % 3x3dofx(n+1)

        for i = GV.n+1:-1:1  % for each collocation point, backward
            u = Uc(:,i);
            a = zeros(3,1);
            b = zeros(3,1);
            A = zeros(3,3);
            %G = zeros(3,3);
            H = zeros(3,3);
            da = zeros(3,3);
            db = zeros(3,3);
            dA = zeros(3,3,3);
            dG = zeros(3,3,3);
            dH = zeros(3,3,3);
            nbt = zeros(3,1);
            dnbtdu = zeros(3,3);
            R = T(1:3,1:3,i); %p = T(1:3,4,i);
            if i == GV.n+1, meL = R'*Me; end

            u_hat = [0 -u(3) u(2);
                u(3) 0 -u(1);
                -u(2) u(1) 0];

            for j = 1 : GV.num_tendons % these are all "local" variables
                pb_si = cross(u,GV.r(:,j)) + [0;0;1];
                pb_s_norm = norm(pb_si);
                Fb_j = -tau(j)*pb_si/pb_s_norm;
                nbt = nbt - Fb_j;
                hat_pb_si = [0 -pb_si(3) pb_si(2);
                    pb_si(3) 0 -pb_si(1);
                    -pb_si(2) pb_si(1) 0];
                ppt = pb_si*pb_si';

                A_j = -(ppt - pb_s_norm^2*eye(3)) * (tau(j)/pb_s_norm^3);
                G_j = -A_j * GV.hat_r(:,:,j);
                a_j = A_j * cross(u,pb_si);

                a = a + a_j;
                b = b + cross(GV.r(:,j), a_j);
                A = A + A_j;
                %G = G + G_j;
                H = H + GV.hat_r(:,:,j)*G_j;

                dpb_si = -GV.hat_r(:,:,j); % 3x3
                Dp = pagemtimes(GV.one3,pb_si'); Dp = Dp + pagetranspose(Dp); % 3x3x3
                dp = 2*pagemtimes(pb_si',GV.one3); % 1x1x3
                dA_j = -tau(j)*((Dp+pagemtimes(dp,eye(3))/2)/pb_s_norm^3 - 3*pagemtimes(dp,ppt)/(2*pb_s_norm^5)); % 3x3x3
                dG_j = -pagemtimes(dA_j, GV.hat_r(:,:,j)); % 3x3x3
                dajdxi = (squeeze(pagemtimes(dA_j, cross(u,pb_si))) + A_j*u_hat)*dpb_si -...
                    A_j*hat_pb_si; % 3x3
                
                dA = dA + tensorprod(dA_j,dpb_si,3,1); % 3x3x3
                da = da + dajdxi; % 3x3
                db = db + GV.hat_r(:,:,j)*dajdxi; % 3x3
                dGjdxi = tensorprod(dG_j,dpb_si,3,1);
                dG = dG + dGjdxi; % 3x3x3
                dH = dH + pagemtimes(GV.hat_r(:,:,j),dGjdxi); % 3x3x3

                dFb_j = -tau(j)*(GV.one3/pb_s_norm - pagemtimes(dp,pb_si)/(2*pb_s_norm^3)); % 3x1x3
                dFbjdu = squeeze(dFb_j)*dpb_si; % 3x3
                dnbtdu = dnbtdu - dFbjdu;

                if i == GV.n+1 % boundary condition
                    meL = meL + cross(GV.r(:,j), Fb_j);
                    dgLu = dgLu + GV.hat_r(:,:,j)*dFbjdu;
                end
            end

            K = H + GV.Kbt; % 3x3
            nb = -nbt + R'*Intfe(:,i); % 3x1 not local
            mb = GV.Kbt*u; % 3x1

            % Calculate ODE terms
            gx(:,i) = -K\(cross(u,mb) + cross([0;0;1],nb) + b); % 3x1

            % Calculate gradient of collocation error
            % dg = -K\(dK*g + drhs)
            % R'*fe, R is coupled with all uc 
            dIntfebdc = squeeze(pagemtimes(dT(1:3,1:3,:,i),'transpose',Intfe(:,i),'none')) + ...
                R'*dIntfedc(:,:,i); % 3x3dof
            
            % u_hat*K*u, not coupled
            mb_hat = [0 -mb(3) mb(2);
                mb(3) 0 -mb(1);
                -mb(2) mb(1) 0];
            
            drhsi = GV.e3_hat*dIntfebdc + ...
                (-GV.e3_hat*dnbtdu + u_hat*GV.Kbt - mb_hat + db)*GV.dUcdc(:,:,i);

            dg(i*3-2:i*3,:) = -K\(drhsi + squeeze(pagemtimes(dH,gx(:,i)))*GV.dUcdc(:,:,i)); % 3x3dof
%             dg(i*3-2:i*3,:) = dgi;
        end
        lambda = [mb;nb];
        
        % Assemble collocation and boundary errors
        bL = GV.Kbt\meL;
        dbL = GV.Kbt\(dgLu*GV.dUcdc(:,:,end) + squeeze(pagemtimes(dT(1:3,1:3,:,end),'transpose',Me,'none')));

        E = Coeffs*[GV.D' GV.S(end,:)'] - [gx bL]; % 3xdof
        e = reshape(E,[],1); % careful here
        J = [GV.dUdotdc;GV.dUcdc(:,:,end)] - [dg;dbL]; % 3dofx3dof

        % max(abs(sum(e.*J,1)))
    end

    function [expPsi,dexp_Psi] = expSE3(Psi)
        % exponential map se3 to SE3, refer to Muller's review
        Psiw = Psi(1:3); Psiv = Psi(4:6);
        theta = norm(Psiw);
        if theta < 1e-16
            expPsiw = eye(3); expPsiv = Psiv;
            dexpPsiw = eye(3);
            y_hat = [0 -Psiv(3) Psiv(2);
                Psiv(3) 0 -Psiv(1);
                -Psiv(2) Psiv(1) 0];
            D_xdexpy = y_hat/2; % (2.36)
        else
            n = Psiw/theta;
            n_hat = [0 -n(3) n(2);
                n(3) 0 -n(1);
                -n(2) n(1) 0];
            n_hat2 = n_hat*n_hat;
            alpha = sin(theta)/theta;
            beta = (1-cos(theta))/(theta^2);
            expPsiw = eye(3) + alpha*n_hat*theta + beta*n_hat2*theta^2; % (2.6)
            dexpPsiw = eye(3) + beta*n_hat*theta + (1-alpha)*n_hat2; % (2.13)
            expPsiv = dexpPsiw*Psiv; % (2.27)

            % derivatives
            delta = (1-alpha)/(theta^2); % (2.4)
            x_hat = n_hat*theta;
            y_hat = [0 -Psiv(3) Psiv(2);
                Psiv(3) 0 -Psiv(1);
                -Psiv(2) Psiv(1) 0];
            D_xdexpy = beta*y_hat + delta*(x_hat*y_hat+y_hat*x_hat) +...
                n'*Psiv*((alpha-2*beta) + (beta-3*delta)*x_hat)*n_hat; % (2.33)
        end
        expPsi = [expPsiw expPsiv;0 0 0 1];
        dexp_Psi = [dexpPsiw zeros(3); D_xdexpy dexpPsiw]; % (2.29), 6x6
    end

    function [dedtau,dedL,dTdL,dnbdc,dnbdL] = collocation_Jacobian(coeffs,L)
        Coeffs = reshape(coeffs,[3 GV.dof]);
        Uc = Coeffs*GV.S'; % 3x(n+1), strain at collocation points

        uq = GV.Q*Coeffs'; % vn x 3, strain at quadrature points
        dPsidB = zeros(6,6,3);
        dTdL = zeros(4,4,GV.n+1);
        dTqdL = zeros(4,4,3*GV.n);

         for k = 1:GV.n % forward kinematics propagation, calculate SE3 transformation and derivatives
            % snapshots at quadrature point
            Xq = [uq(k*3-2:k*3,:)  repmat([0 0 1],3,1)]; % vx6

            % Calculate Magnus expansion
            B = (GV.invV*Xq*GV.hsp(k))'; % 6xv self-adjoint basis
            B1 = B(:,1); B2 = B(:,2); B3 = B(:,3);
            w1_hat = [0 -B1(3) B1(2);
                B1(3) 0 -B1(1);
                -B1(2) B1(1) 0];
            v1_hat = [0 -B1(6) B1(5);
                B1(6) 0 -B1(4);
                -B1(5) B1(4) 0];
            adB1 = [w1_hat zeros(3);v1_hat w1_hat];
            w2_hat = [0 -B2(3) B2(2);
                B2(3) 0 -B2(1);
                -B2(2) B2(1) 0];
            v2_hat = [0 -B2(6) B2(5);
                B2(6) 0 -B2(4);
                -B2(5) B2(4) 0];
            adB2 = [w2_hat zeros(3);v2_hat w2_hat];
            w3_hat = [0 -B3(3) B3(2);
                B3(3) 0 -B3(1);
                -B3(2) B3(1) 0];
            v3_hat = [0 -B3(6) B3(5);
                B3(6) 0 -B3(4);
                -B3(5) B3(4) 0];
            adB3 = [w3_hat zeros(3);v3_hat w3_hat];

            B12 = adB1*B2;
            B23 = adB2*B3;
            B13 = adB1*B3;
            B113 = adB1*B13;
            B212 = adB2*B12;
            B112 = adB1*B12;
            B1112 = adB1*B112;

            Psi = B1 + B3/12 + B12/12 - B23/240 + B113/360 - B212/240 - B1112/720;

            % exponential map se3 to SE3
            [expPsi,dexp_Psi] = expSE3(Psi);

            % Integrate on SE3
            % T(:,:,k+1) = T(:,:,k)*expPsi;

            % assume constant curvature between quadrature points
            Psiq1 = (Xq(1,:)' + [Uc(:,k);0;0;1])/2*GV.hsp(k)*(1/2-sqrt(15)/10);
            Psiq2 = (Xq(2,:)' + Xq(1,:)')/2*GV.hsp(k)*sqrt(15)/10;
            Psiq3 = (Xq(3,:)' + [Uc(:,k+1);0;0;1])/2*GV.hsp(k)*(1/2-sqrt(15)/10);
            [expPsiq1,dexp_Psiq1] = expSE3(Psiq1);
            [expPsiq2,dexp_Psiq2] = expSE3(Psiq2);
            [expPsiq3,dexp_Psiq3] = expSE3(Psiq3);
            % Tq(:,:,k*3-2) = T(:,:,k)*expPsiq1;
            % Tq(:,:,k*3-1) = Tq(:,:,k*3-2)*expPsiq2;
            % Tq(:,:,k*3) = T(:,:,k+1)/expPsiq3;


            % Derivatives on SE3
            % from uc to uq
            % duqdc = zeros(3,3*GV.dof,3);
%             duqdc = GV.dUqdc(:,:,k); % 3vx3dof

            % from uq to B
%             dXqduq = GV.hsp(k)*[eye(3);zeros(3)]; % 6x3
            dXqdL = Xq'*GV.hsp(k)/L; % 6xv

            % from B to Psi
            w12_hat = [0 -B12(3) B12(2);
                B12(3) 0 -B12(1);
                -B12(2) B12(1) 0];
            v12_hat = [0 -B12(6) B12(5);
                B12(6) 0 -B12(4);
                -B12(5) B12(4) 0];
            adB12 = [w12_hat zeros(3);v12_hat w12_hat];
            w13_hat = [0 -B13(3) B13(2);
                B13(3) 0 -B13(1);
                -B13(2) B13(1) 0];
            v13_hat = [0 -B13(6) B13(5);
                B13(6) 0 -B13(4);
                -B13(5) B13(4) 0];
            adB13 = [w13_hat zeros(3);v13_hat w13_hat];
            w112_hat = [0 -B112(3) B112(2);
                B112(3) 0 -B112(1);
                -B112(2) B112(1) 0];
            v112_hat = [0 -B112(6) B112(5);
                B112(6) 0 -B112(4);
                -B112(5) B112(4) 0];
            adB112 = [w112_hat zeros(3);v112_hat w112_hat];
            dPsidB(:,:,1) = eye(6) - adB2/12 - (adB1*adB3 + adB13)/360 +...
                adB2*adB2/240 + (adB1*adB1*adB2 + adB1*adB12 + adB112)/720; % 6x6
            dPsidB(:,:,2) = adB1/12 + adB3/240 + (-adB2*adB1 + adB12)/240 -...
                adB1*adB1*adB1/720; % 6x6
            dPsidB(:,:,3) = eye(6)/12 - adB2/240 + adB1*adB1/360; % 6x6

            % from Psi to dR
%             dPsidc = zeros(6,3*GV.dof);
            dPsidL = zeros(6,1);
            for i = 1:3
                for j = 1:3
                    % derivative through Bi to Aj (uqj)
%                     dPsidc = dPsidc + dPsidB(:,:,i)*GV.invV(i,j)*dXqduq*duqdc(j*3-2:j*3,:);
                    dPsidL = dPsidL + dPsidB(:,:,i)*GV.invV(i,j)*dXqdL(:,j);
                end
            end
            D = dexp_Psi*dPsidL; % 6x1
            D_hat = [0 -D(3) D(2) D(4);
                    D(3) 0 -D(1) D(5);
                    -D(2) D(1) 0 D(6);
                    0 0 0 0];

%             dT(:,:,:,k+1) = pagemtimes(dT(:,:,:,k) + pagemtimes(T(:,:,k),D_hat),expPsi); % 4x4x3dof
            dTdL(:,:,k+1) = (dTdL(:,:,k) + T(:,:,k)*D_hat)*expPsi; % 4x4
            
            dPsiq1dL = Psiq1/L; % 6x1
            Dq1 = dexp_Psiq1*dPsiq1dL; % 6x1
                Dq1_hat = [0 -Dq1(3) Dq1(2) Dq1(4);
                    Dq1(3) 0 -Dq1(1) Dq1(5);
                    -Dq1(2) Dq1(1) 0 Dq1(6);
                    0 0 0 0];
            dTqdL(:,:,k*3-2) = (dTdL(:,:,k) + T(:,:,k)*Dq1_hat)*expPsiq1;
            dPsiq2dL = Psiq2/L; % 6x1
            Dq2 = dexp_Psiq2*dPsiq2dL; % 6x1
                Dq2_hat = [0 -Dq2(3) Dq2(2) Dq2(4);
                    Dq2(3) 0 -Dq2(1) Dq2(5);
                    -Dq2(2) Dq2(1) 0 Dq2(6);
                    0 0 0 0];
            dTqdL(:,:,k*3-1) = (dTqdL(:,:,k*3-2) + Tq(:,:,k*3-2)*Dq2_hat)*expPsiq2;
            dPsiq3dL = Psiq3/L; % 6x1
            Dq3 = dexp_Psiq3*dPsiq3dL; % 6x1
                Dq3_hat = [0 -Dq3(3) Dq3(2) Dq3(4);
                    Dq3(3) 0 -Dq3(1) Dq3(5);
                    -Dq3(2) Dq3(1) 0 Dq3(6);
                    0 0 0 0];
            dTqdL(:,:,k*3) = dTdL(:,:,k+1)/expPsiq3 - Tq(:,:,k*3)*Dq3_hat;
        end

        % Setup tendon linear system
        gx = zeros(3,GV.n+1);
        dgxdtau = zeros(3*(GV.n+1),GV.num_tendons);
        dgdL = zeros(3*(GV.n+1),1);
        % nbt = zeros(3,GV.n+1); % internal actuation force
        % dgLu = squeeze(pagemtimes(dT(1:3,1:3,:,end),'transpose',Me,'none'));
%         dgLu = zeros(3);

        % quadrature for integration of distributed external force
        [fei,dfei] = fe(squeeze(Tq(1:3,4,:))); dfei = squeeze(dfei); % 3x3xvn
        intfe = [GV.hsp.*(5*(fei(:,1:3:end) + fei(:,3:3:end)) + 8*fei(:,2:3:end))/18 Fe];
        Intfe = cumsum(intfe,2,"reverse");

        dfedc = pagemtimes(dfei,squeeze(dTq(1:3,4,:,:))); % 3x3dofxvn
        dintfedc = cat(3,...
            reshape(GV.hsp,1,1,[]).*(5*(dfedc(:,:,1:3:end) + dfedc(:,:,3:3:end)) + 8*dfedc(:,:,2:3:end))/18,...
            zeros(3,3*GV.dof)); % 3x3dofx(n+1)
        dIntfedc = cumsum(dintfedc,3,"reverse"); % 3x3dofx(n+1)

        dfedL = squeeze(pagemtimes(dfei,dTqdL(1:3,4,:))); % 3xvn
        dintfedL = [GV.hsp.*(5*(dfedL(:,1:3:end)+dfedL(:,3:3:end))+8*dfedL(:,2:3:end))/18+intfe(:,1:end-1)/L zeros(3,1)]; % 3x(n+1)
        dIntfedL = cumsum(dintfedL,2,"reverse"); % 3x(n+1)

        dmeL = zeros(3,GV.num_tendons);
        dnbtdu = zeros(3,3);
        for i = GV.n+1:-1:1  % for each collocation point, backward
            u = Uc(:,i);
            a = zeros(3,1);
            b = zeros(3,1);
            A = zeros(3,3);
            %G = zeros(3,3);
            H = zeros(3,3);

            dadtau = zeros(3,GV.num_tendons);
            dbdtau = zeros(3,GV.num_tendons);
            dAdtau = zeros(3,3,GV.num_tendons);
            dGdtau = zeros(3,3,GV.num_tendons);
            dHdtau = zeros(3,3,GV.num_tendons);

            nbt = zeros(3,1);
            dnbtdtau = zeros(3,GV.num_tendons);
            R = T(1:3,1:3,i); %p = T(1:3,4,i);
            if i == GV.n+1, meL = R'*Me; end

%             u_hat = [0 -u(3) u(2);
%                 u(3) 0 -u(1);
%                 -u(2) u(1) 0];

            for j = 1 : GV.num_tendons % these are all "local" variables
                pb_si = cross(u,GV.r(:,j)) + [0;0;1];
                pb_s_norm = norm(pb_si);
                dFbjdtau = -pb_si/pb_s_norm;
                Fb_j = tau(j)*dFbjdtau;
                nbt = nbt - Fb_j;
%                 hat_pb_si = [0 -pb_si(3) pb_si(2);
%                     pb_si(3) 0 -pb_si(1);
%                     -pb_si(2) pb_si(1) 0];
                ppt = pb_si*pb_si';

                A_j = -(ppt - pb_s_norm^2*eye(3)) * (tau(j)/pb_s_norm^3);
                G_j = -A_j * GV.hat_r(:,:,j);
                a_j = A_j * cross(u,pb_si);

                a = a + a_j;
                b = b + cross(GV.r(:,j), a_j);
                A = A + A_j;
                %G = G + G_j;
                H = H + GV.hat_r(:,:,j)*G_j;

                dAdtau(:,:,j) = -(ppt - pb_s_norm^2*eye(3))/(pb_s_norm^3);
                dGdtau(:,:,j) = -dAdtau(:,:,j) * GV.hat_r(:,:,j);
                dadtau(:,j) = dAdtau(:,:,j) * cross(u,pb_si);
                dbdtau(:,j) = cross(GV.r(:,j), dadtau(:,j));
                dHdtau(:,:,j) = GV.hat_r(:,:,j)*dGdtau(:,:,j);

                dnbtdtau(:,j) = -dFbjdtau; %!!!

                if i == GV.n+1 % boundary condition
%                     meL = meL + cross(GV.r(:,j), Fb_j);
%                     dgLu = dgLu + GV.hat_r(:,:,j)*dFbjdu;
                    dmeL(:,j) = cross(GV.r(:,j), dFbjdtau);
                end

                if i == 1
                    dpb_si = -GV.hat_r(:,:,j); % 3x3
                    dp = 2*pagemtimes(pb_si',GV.one3); % 1x1x3
                    dFb_j = -tau(j)*(GV.one3/pb_s_norm - pagemtimes(dp,pb_si)/(2*pb_s_norm^3)); % 3x1x3
                    dFbjdu = squeeze(dFb_j)*dpb_si; % 3x3
                    dnbtdu = dnbtdu - dFbjdu;
                end
            end

            K = H + GV.Kbt; % 3x3
            dKdtau = dHdtau;
            nb = -nbt + R'*Intfe(:,i); % 3x1 not local
            dnbdtau = -dnbtdtau;
            dnbdL = dTdL(1:3,1:3,i)'*Intfe(:,i) + R'*dIntfedL(:,i); % 3x1
            mb = GV.Kbt*u; % 3x1

            % Calculate ODE terms
            gx(:,i) = -K\(cross(u,mb) + cross([0;0;1],nb) + b); % 3x1
            drhsdtau = GV.e3_hat*dnbdtau + dbdtau; % 3xnum_tendon
            dgxdtau(3*(i-1)+1:3*(i-1)+3,:) = -K\(squeeze(pagemtimes(dKdtau,gx(:,i))) + drhsdtau);
            
            drhsidL = GV.e3_hat*dnbdL;
            dgdL(i*3-2:i*3,:) = -K\drhsidL;

            % Calculate gradient of collocation error
            % dg = -K\(dK*g + drhs)
            % R'*fe, R is coupled with all uc 
        end
        dIntfebdc = squeeze(pagemtimes(dT(1:3,1:3,:,1),'transpose',Intfe(:,1),'none')) + ...
                R'*dIntfedc(:,:,1); % 3x3dof
        dnbdc = -dnbtdu*GV.dUcdc(:,:,1) + dIntfebdc; % 3x3dof for base force
        
        % Assemble collocation and boundary errors
%         bL = GV.Kbt\meL;
%         dbL = GV.Kbt\(dgLu*GV.dUcdc(:,:,end) + squeeze(pagemtimes(dT(1:3,1:3,:,end),'transpose',Me,'none')));
        dbLdtau = GV.Kbt\dmeL;

%         E = Coeffs*[GV.D' GV.S(end,:)'] - [gx bL]; % 3xdof
%         e = reshape(E,[],1); % careful here
        dedtau = -[dgxdtau;dbLdtau];
        dedL = reshape(Coeffs*[-GV.D'/L zeros(GV.dof,1)],[],1) - [dgdL;zeros(3,1)];%!
    end

    function dedL = collocation_Jacobian_L(coeffs,L)
        % coeffs: 3dofx1 spline coefficients stacked
        % Coeffs: 3xdof spline coefficients
        % calculate pose at collocation points via magnus expansion
        Coeffs = reshape(coeffs,[3 GV.dof]);
        uq = GV.Q*Coeffs'; % vn x 3, strain at quadrature points
        Uc = Coeffs*GV.S'; % 3x(n+1), strain at collocation points
        dPsidB = zeros(6,6,3);
        dTdL = zeros(4,4,GV.n+1);
        dTqdL = zeros(4,4,3*GV.n);

        for k = 1:GV.n % forward kinematics propagation, calculate SE3 transformation and derivatives
            % snapshots at quadrature point
            Xq = [uq(k*3-2:k*3,:)  repmat([0 0 1],3,1)]; % vx6

            % Calculate Magnus expansion
            B = (GV.invV*Xq*GV.hsp(k))'; % 6xv self-adjoint basis
            B1 = B(:,1); B2 = B(:,2); B3 = B(:,3);
            w1_hat = [0 -B1(3) B1(2);
                B1(3) 0 -B1(1);
                -B1(2) B1(1) 0];
            v1_hat = [0 -B1(6) B1(5);
                B1(6) 0 -B1(4);
                -B1(5) B1(4) 0];
            adB1 = [w1_hat zeros(3);v1_hat w1_hat];
            w2_hat = [0 -B2(3) B2(2);
                B2(3) 0 -B2(1);
                -B2(2) B2(1) 0];
            v2_hat = [0 -B2(6) B2(5);
                B2(6) 0 -B2(4);
                -B2(5) B2(4) 0];
            adB2 = [w2_hat zeros(3);v2_hat w2_hat];
            w3_hat = [0 -B3(3) B3(2);
                B3(3) 0 -B3(1);
                -B3(2) B3(1) 0];
            v3_hat = [0 -B3(6) B3(5);
                B3(6) 0 -B3(4);
                -B3(5) B3(4) 0];
            adB3 = [w3_hat zeros(3);v3_hat w3_hat];

            B12 = adB1*B2;
            B23 = adB2*B3;
            B13 = adB1*B3;
            B113 = adB1*B13;
            B212 = adB2*B12;
            B112 = adB1*B12;
            B1112 = adB1*B112;

            Psi = B1 + B3/12 + B12/12 - B23/240 + B113/360 - B212/240 - B1112/720;

            % exponential map se3 to SE3
            [expPsi,dexp_Psi] = expSE3(Psi);

            % Integrate on SE3
            T(:,:,k+1) = T(:,:,k)*expPsi;

            % assume constant curvature between quadrature points
            Psiq1 = (Xq(1,:)' + [Uc(:,k);0;0;1])/2*GV.hsp(k)*(1/2-sqrt(15)/10);
            Psiq2 = (Xq(2,:)' + Xq(1,:)')/2*GV.hsp(k)*sqrt(15)/10;
            Psiq3 = (Xq(3,:)' + [Uc(:,k+1);0;0;1])/2*GV.hsp(k)*(1/2-sqrt(15)/10);
            [expPsiq1,dexp_Psiq1] = expSE3(Psiq1);
            [expPsiq2,dexp_Psiq2] = expSE3(Psiq2);
            [expPsiq3,dexp_Psiq3] = expSE3(Psiq3);
            Tq(:,:,k*3-2) = T(:,:,k)*expPsiq1;
            Tq(:,:,k*3-1) = Tq(:,:,k*3-2)*expPsiq2;
            Tq(:,:,k*3) = T(:,:,k+1)/expPsiq3;


            % Derivatives on SE3
            % from uc to uq
            % duqdc = zeros(3,3*GV.dof,3);
%             duqdc = GV.dUqdc(:,:,k); % 3vx3dof

            % from uq to B
%             dXqduq = GV.hsp(k)*[eye(3);zeros(3)]; % 6x3
            dXqdL = Xq*GV.hsp(k)/L;

            % from B to Psi
            w12_hat = [0 -B12(3) B12(2);
                B12(3) 0 -B12(1);
                -B12(2) B12(1) 0];
            v12_hat = [0 -B12(6) B12(5);
                B12(6) 0 -B12(4);
                -B12(5) B12(4) 0];
            adB12 = [w12_hat zeros(3);v12_hat w12_hat];
            w13_hat = [0 -B13(3) B13(2);
                B13(3) 0 -B13(1);
                -B13(2) B13(1) 0];
            v13_hat = [0 -B13(6) B13(5);
                B13(6) 0 -B13(4);
                -B13(5) B13(4) 0];
            adB13 = [w13_hat zeros(3);v13_hat w13_hat];
            w112_hat = [0 -B112(3) B112(2);
                B112(3) 0 -B112(1);
                -B112(2) B112(1) 0];
            v112_hat = [0 -B112(6) B112(5);
                B112(6) 0 -B112(4);
                -B112(5) B112(4) 0];
            adB112 = [w112_hat zeros(3);v112_hat w112_hat];
            dPsidB(:,:,1) = eye(6) - adB2/12 - (adB1*adB3 + adB13)/360 +...
                adB2*adB2/240 + (adB1*adB1*adB2 + adB1*adB12 + adB112)/720; % 6x6
            dPsidB(:,:,2) = adB1/12 + adB3/240 + (-adB2*adB1 + adB12)/240 -...
                adB1*adB1*adB1/720; % 6x6
            dPsidB(:,:,3) = eye(6)/12 - adB2/240 + adB1*adB1/360; % 6x6

            % from Psi to dR
%             dPsidc = zeros(6,3*GV.dof);
            dPsidL = zeros(6,1);
            for i = 1:3
                for j = 1:3
                    % derivative through Bi to Aj (uqj)
%                     dPsidc = dPsidc + dPsidB(:,:,i)*GV.invV(i,j)*dXqduq*duqdc(j*3-2:j*3,:);
                    dPsidL = dPsidc + dPsidB(:,:,i)*GV.invV(i,j)*dXqdL;
                end
            end
            D = dexp_Psi*dPsidL; % 6x1
            D_hat = [0 -D(3) D(2) D(4);
                    D(3) 0 -D(1) D(5);
                    -D(2) D(1) 0 D(6);
                    0 0 0 0];

%             dT(:,:,:,k+1) = pagemtimes(dT(:,:,:,k) + pagemtimes(T(:,:,k),D_hat),expPsi); % 4x4x3dof
            dTdL(:,:,k+1) = (dTdL(:,:,k) + T(:,:,k)*D_hat)*expPsi; % 4x4
            
            dPsiq1dL = Psiq1/L; % 6x1
            Dq1 = dexp_Psiq1*dPsiq1dL; % 6x1
                Dq1_hat = [0 -Dq1(3) Dq1(2) Dq1(4);
                    Dq1(3) 0 -Dq1(1) Dq1(5);
                    -Dq1(2) Dq1(1) 0 Dq1(6);
                    0 0 0 0];
            dTqdL(:,:,k*3-2) = (dTdL(:,:,k) + T(:,:,k)*Dq1_hat)*expPsiq1;
            dPsiq2dL = Psiq2/L; % 6x1
            Dq2 = dexp_Psiq2*dPsiq2dL; % 6x1
                Dq2_hat = [0 -Dq2(3) Dq2(2) Dq2(4);
                    Dq2(3) 0 -Dq2(1) Dq2(5);
                    -Dq2(2) Dq2(1) 0 Dq2(6);
                    0 0 0 0];
            dTqdL(:,:,k*3-1) = (dTqdL(:,:,k*3-2) + Tq(:,:,k*3-2)*Dq2_hat)*expPsiq2;
            dPsiq3dL = Psiq3/L; % 6x1
            Dq3 = dexp_Psiq3*dPsiq3dL; % 6x1
                Dq3_hat = [0 -Dq3(3) Dq3(2) Dq3(4);
                    Dq3(3) 0 -Dq3(1) Dq3(5);
                    -Dq3(2) Dq3(1) 0 Dq3(6);
                    0 0 0 0];
            dTqdL(:,:,k*3) = dTdL(:,:,k+1)/expPsiq3 - Tq(:,:,k*3)*Dq3_hat;
        end

        % Setup tendon linear system
%         gx = zeros(3,GV.n+1);
        dgdL = zeros(3*(GV.n+1),1);
        % quadrature for integration of distributed external force
        [fei,dfei] = fe(squeeze(Tq(1:3,4,:))); dfei = squeeze(dfei); % 3x3xvn
        intfe = [GV.hsp.*(5*(fei(:,1:3:end) + fei(:,3:3:end)) + 8*fei(:,2:3:end))/18 Fe]; % 3xn
        Intfe = cumsum(intfe,2,"reverse");
        dfedL = squeeze(pagemtimes(dfei,dTqdL(1:3,4,:))); % 3xvn
        dintfedL = [GV.hsp.*(5*(dfedL(:,1:3:end)+dfedL(:,3:3:end))+8*dfedL(:,2:3:end))/18+intfe/L zeros(3,3*GV.dof)]; % 3x(n+1)
        dIntfedL = cumsum(dintfedL,3,"reverse"); % 3x(n+1)

        for i = GV.n+1:-1:1  % for each collocation point, backward
            u = Uc(:,i);
            a = zeros(3,1);
            b = zeros(3,1);
            A = zeros(3,3);
            %G = zeros(3,3);
            H = zeros(3,3);
            nbt = zeros(3,1);
            R = T(1:3,1:3,i); %p = T(1:3,4,i);

            for j = 1 : GV.num_tendons % these are all "local" variables
                pb_si = cross(u,GV.r(:,j)) + [0;0;1];
                pb_s_norm = norm(pb_si);
                Fb_j = -tau(j)*pb_si/pb_s_norm;
                nbt = nbt - Fb_j;
                ppt = pb_si*pb_si';

                A_j = -(ppt - pb_s_norm^2*eye(3)) * (tau(j)/pb_s_norm^3);
                G_j = -A_j * GV.hat_r(:,:,j);
                a_j = A_j * cross(u,pb_si);

                a = a + a_j;
                b = b + cross(GV.r(:,j), a_j);
                A = A + A_j;
                %G = G + G_j;
                H = H + GV.hat_r(:,:,j)*G_j;
            end

            K = H + GV.Kbt; % 3x3
%             nb = -nbt + R'*Intfe(:,i); % 3x1 not local
            dnbdL = dTdL(1:3,1:3,i)'*Intfe(:,i) + R'*dIntfedL(:,i); % 3x1
%             mb = GV.Kbt*u; % 3x1

            % Calculate ODE terms
%             gx(:,i) = -K\(cross(u,mb) + cross([0;0;1],nb) + b); % 3x1

            drhsidL = GV.e3_hat*dnbdL;
            dgdL(i*3-2:i*3,:) = -K\drhsidL;
        end
        dedL = reshape(Coeffs*[-GV.D'/L zeros(3,1)] - [dgdL zeros(3,1)],[],1);
    end
end