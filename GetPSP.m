%  v=GetPSP(Cm,tauf,taur,J,gl,V0,T,dt)
% Compute one PSP.  
% tauf is PSC decay time constant.
% taur is rise time cnst (set to 0 for exponential PSC
% J is synaptic weight
% Cm, gl are capacitance and leak for LIF
% V0 is initial potential
% Solves:
% Cm*V'==-gL*(V-EL)+I
% where I==alphaf-alphar and
% tauf*alphaf'=-alphaf
% taur*alphar'=-alphar
% with I(0)==J/(alphaf-alphar) so that the integral of I is J
% and with V(0)==V0

function v=GetPSP(Cm,tauf,taur,J,gl,V0,T,dt)

    Nt=round(T/dt);
    v=zeros(Nt,1);
    alphaf=zeros(Nt,1);
    alphar=zeros(Nt,1);
    v(1)=V0;

    % If rise time is zero, only account for decay
    if(taur<1e-9)
        alphaf(1)=J/tauf;
        for i=2:Nt
            alphaf(i)=alphaf(i-1)-dt*alphaf(i-1)/tauf;
            v(i)=v(i-1)+dt/Cm*(alphaf(i)-gl*(v(i-1)-V0));
        end       
    else  % Otherwise, account for rise too
        alphar(1)=J/(tauf-taur);
        for i=2:Nt
            alphaf(i)=alphaf(i-1)-dt*alphaf(i-1)/tauf;
            alphar(i)=alphar(i-1)-dt*alphar(i-1)/taur;
            v(i)=v(i-1)+dt/Cm*((alphaf(i)-alphar(i))-gl*(v(i-1)-V0));
        end    
    end



end
