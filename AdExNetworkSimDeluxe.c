/*
[s,Ix,Ie,Ii,v,w,noise,mIx,mIe,mIi,mv,mw]=AdExNetworkSimDeluxe(N,Nx,NeuronType,T,sx,Istim,JIstim,Jinds,Jweights, ...
                                                              J0,Jxinds,Jxweights,Jx0,NeuronParams,taux,taue,taui, ...
                                                              v0,dt,maxns,Tburn,sigma,Irecord,navgrecord)
 
 Simulate a recurrent network of AdEx neuron models receiving ffwd input from an external layer of spike trains.
 
 The neuron model satisfies:
 	Cm*V'(t)=-gL(V-EL)+Iapp+gL*DeltaT*exp((V-VT)/DeltaT)-w+Isyn+noise
 	tauw*w'(t)=-w+a*(V-EL)
 with a lower bound on V at Vlb and a threshold at Vth.
 Every time the neuron crosses threshold, a spike is recorded, it is held for tauref,
 and w is incremented by b.
 To get an EIF model, set a=0 and b=0.
 To get an LIF model, set a=0, b=0, DeltaT=0, VT>Vth
 
 
 Inputs:
 N is the number of neurons in the recurrent net
 
 Nx is the number of neurons in the ffwd layer. If you don't want a ffwd layer, set Nx=0
 
 NeuronType is a Nx1 vector indicating whether each neuron is excitatory (0)
 or inhibitory (1).
 
 T is the length of time for the simulation (in ms)
 
 sx is a 2xNsx vector of spike times for the ffwd layer. 
 sx(1,:) contains spike times in ascending order
 sx(2,:) contains neuron indices (from 1 to Nx)
 If you don't want a ffwd layer, set sx=[]
 
 Istim is an Ntx1 vector of with a time-dependent stimulus (Nt=round(T/dt)) 
 JIstim is a Nx1 vector of associated weights.
 At time bin i, neuron j gets extra additive input Istim(i)*JIstim(j) 
 If you don't want this extra input, set Istim=[], JIstim=[]
 
 Jinds contains postsynaptic indices to/from all N neurons in the recurrent net
 Jweights are the associated synaptic weights
 J0 is a (N+1)x1 vector of pointers into Jinds and Jweights
 The postsynaptic indices for presynaptic neuron k are stored in Jinds(J0(k):J0(k+1))
 and the weights are in Jweights(J0(k):J0(k+1))
 If you don't want recurrent connectivity, set all three to [].
 
 Jxinds, Jxweights, Jx0 are similar, but for presynaptic neurons in the ffwd layer.
 If you don't want a ffwd layer, set all to [].
  
 NeuronParams is a 2x13 vector of AdEx neuron parameters.
 NeuronParams(1,:) is for excitatory neurons and NeuronParams(2,:) for inhibitory
 The params are (in order): Cm gL EL Vth Vre Vlb tauref Iapp DeltaT VT a b tauw

 taux, taue, taui are the synaptic time constants for ffwd, excitatory, and inhibitory
 synapses. PSCs are of the form (1/taub)*exp(-(t-tspike)/taub)*(t>tspike)
 These generate Isyn.
 
 v0 is an Nx1 vector of initial membrane potentials
 
 dt is the time step
 
 maxns is the maximum number of spikes. If more spikes occur, the program will 
 terminate with a warning.
 
 Tburn is a burn-in period for computing averages (see mIx, mIi, mIe outputs)
 
 sigma is an Nx1 vector of noise coefficients. Specifically, the noise term is 
 sigma*eta(t) where eta(t) is standard Gaussian white noise.
 
 Irecord is a vector of neuron indices for which to record synaptic currents, 
 membrane potentials, noise, and w (see outputs). 
 The indices must be in ascending order.
 If you don't want to record these for any neurons, set Irecord=[].
 
 navgrecord is the number of time bins over which recorded variables are averaged. 
 To record every time bin, set navgrecord=1.
 If you set navgrecord=10, for example, then variables are recorded on a
 coarser time mesh of dt*10. This uses 10 times less memory.
 
 
 Outputs:
 
 sx is a 2xNs vector of spike times 
 s(1,:) contains spike times in ascending order
 s(2,:) contains neuron indices (from 1 to N)
 
 Ix, Ie, Ii, v, w, noise are synaptic input currents (external, excitatory, inhibitory), 
 membrane potential, adaptation current, and noise for recorded neurons (see Irecord input)
 
 m* are the time-averaged quantities for all neurons (so they are Nx1 vectors regardless 
 of what is in Irecord)
 
 */


#define _USE_MATH_DEFINES
#include "mex.h"
#include "math.h"

/* Macros for drand48 */
#ifndef srand48
# define srand48(s) srand(s)
#endif
#ifndef drand48
# define drand48() (((double)rand())/((double)RAND_MAX))
#endif


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int i,ii,j,jj,k,n,Nt,Ntref,maxns,Nf,*J0,*Jf0,Nsf,nJ,m1,m2,N,nJf,*refstate,ns,Ntrefe,Ntrefi,printfperiod,iFspike,jf,*NeuronType,m1Istim,m2Istim,nburn,*Irecord,Nrecord,navgrecord,IsNoise,Ntrecord,IsStim,jrecord,isF,IsRec;
    double *s,*malphafr,*malphaer,*malphair,*mvr,*mwr,*alphafr,*alphaer,*alphair,*vr,*wr,*noiser,*sf,*Jweights,*Jfweights,Ce,Ci,gLe,gLi,ELe,ELi,DeltaTe,DeltaTi,VTe,VTi,taurefe,taurefi,Vthe,Vthi,Vree,Vrei,Vlbe,Vlbi,tauf,taue,taui,taujitter,*v0,T,dt,Iappe,Iappi,*NeuronParams,*temp,*JnextE,*JnextI,*JnextF,*alphae,*alphai,*alphaf,*v,ae,ai,be,bi,tauwe,tauwi,*w,*Istim,*JIstim,*Jinds,*Jfinds,Tburn,u1,u2,noise,noise1,*sigma,tempstim,tempnoise; 
    
    
    /******
     * Import variables from matlab
     * This is messy looking and is specific to mex.
     *******/

    N=(int)round(mxGetScalar(prhs[0]));    
    Nf=(int)round(mxGetScalar(prhs[1]));
    if(Nf==0)
        isF=0;
    else
        isF=1;
    
    NeuronType=mxMalloc(N*sizeof(int));
    temp=mxGetPr(prhs[2]);
    m1 = mxGetM(prhs[2]);
    m2 = mxGetN(prhs[2]);    
    if(m1*m2!=N)
        mexErrMsgTxt("NeuronType should be Nx1 or 1xN");        
    for(j=0;j<N;j++){
        NeuronType[j]=(int)round(temp[j]);
        if(NeuronType[j]!=0 && NeuronType[j]!=1)
            mexErrMsgTxt("All NeuronTypes should be 0 or 1");        
    }
    
    T=mxGetScalar(prhs[3]);
    
    sf = mxGetPr(prhs[4]);
    m1 = mxGetM(prhs[4]);
    Nsf = mxGetN(prhs[4]); /* number of ffwd spikes */
    if(Nsf*m1!=0 && isF){
        isF=1;
        if(m1!=2){
            mexPrintf("\n%d\n",m1);
            mexErrMsgTxt("sf should be (Nsf)x2");        
        }
        /* Convert sf to C-based indexing and
           check that times and neuron indices are okay */
        sf[0*2+1]=sf[0*2+1]-1;
        if(sf[0*2]<0 || sf[0*2]>T || sf[0*2+1]<0 || sf[0*2+1]>Nf-1)
            mexErrMsgTxt("Out of bounds indices in sf initial.");
        for(j=1;j<Nsf;j++){
            sf[j*2+1]=sf[j*2+1]-1;
            if(sf[j*2]<0 || sf[j*2]>T || sf[j*2+1]<0 || sf[j*2+1]>Nf-1)
                mexErrMsgTxt("Out of bounds indices in sf.");
            if(sf[j*2]<sf[(j-1)*2])            
                mexErrMsgTxt("Spike times in sf must be non-decreasing.");        
        }
    }
    else
        isF=0;
    
    if(Nsf*m1!=0 && Nf==0)
        mexWarnMsgTxt("You set Nf=0, but passed in ffwd spikes. Those spikes will be ignored.");
        
    
    Istim = mxGetPr(prhs[5]);
    m1Istim = mxGetM(prhs[5]);
    m2Istim = mxGetN(prhs[5]); /* number of ffwd currents */    
    if(m1Istim*m2Istim==0)
        IsStim=0;
    else
        IsStim=1;
            
    JIstim=mxGetPr(prhs[6]);
    m1 = mxGetM(prhs[6]);
    m2 = mxGetN(prhs[6]); 
    if(m1*m2!=N && IsStim)
       mexErrMsgTxt("JIstim should be Nx1 or 1xN when Istim is nonempty.");

    
    
    
    Jinds = mxGetPr(prhs[7]);
    m1 = mxGetM(prhs[7]);
    m2 = mxGetN(prhs[7]);
    nJ = m1*m2;
    if(nJ>0)
        IsRec=1;
    else
        IsRec=0;
  
    Jweights = mxGetPr(prhs[8]);
    m1 = mxGetM(prhs[8]);
    m2 = mxGetN(prhs[8]);
    if(m1*m2!=nJ && IsRec)
        mexErrMsgTxt("Jweights must be same length as Jinds");
    
    
    J0=mxMalloc((N+1)*sizeof(int));    
    temp = mxGetPr(prhs[9]);
    m1 = mxGetM(prhs[9]);
    m2 = mxGetN(prhs[9]); 
    if(IsRec){
        if(m1*m2!=N+1)
            mexErrMsgTxt("J0 should ne Nx1 or 1xN.");        
        for(j=0;j<N+1;j++){
            J0[j]=(int)(round(temp[j]-1));
            if(J0[j]<0 || J0[j]>nJ)
                mexErrMsgTxt("Out of bounds indices in J0.");

        }
    }


    Jfinds = mxGetPr(prhs[10]);
    m1 = mxGetM(prhs[10]);
    m2 = mxGetN(prhs[10]); 
    nJf = m1*m2;
    if(nJf>0 && isF)
        isF=1;
    else
        isF=0;
    
   
    Jfweights = mxGetPr(prhs[11]);
    m1 = mxGetM(prhs[11]);
    m2 = mxGetN(prhs[11]);
    if(m1*m2!=nJf && isF)
        mexErrMsgTxt("Jfweights must be same length as Jfinds");
    
    
    Jf0=mxMalloc((Nf+1)*sizeof(int));
    temp = mxGetPr(prhs[12]);
    m1 = mxGetM(prhs[12]);
    m2 = mxGetN(prhs[12]); 
    if(isF){
        if(m1*m2!=Nf+1)
            mexErrMsgTxt("Jf0 should ne (Nf+1)x1 or 1x(Nf+1).");        
        for(j=0;j<Nf+1;j++){
            Jf0[j]=(int)(round(temp[j]-1));
            if(Jf0[j]<0 || Jf0[j]>nJf){
                mexPrintf("\n%d %d\n",j,Jf0[j]);
                mexErrMsgTxt("Out of bounds indices Jf0.");
            }
        }
    }
    

    
    NeuronParams=mxGetPr(prhs[13]);
    m1 = mxGetM(prhs[13]);
    m2 = mxGetN(prhs[13]); 
    if(m1!=2 || m2!=13)
        mexErrMsgTxt("NeuronParams should be 13x2.");
    Ce=NeuronParams[0*2+0];
    Ci=NeuronParams[0*2+1];
    gLe=NeuronParams[1*2+0];
    gLi=NeuronParams[1*2+1];
    ELe=NeuronParams[2*2+0];
    ELi=NeuronParams[2*2+1];
    Vthe=NeuronParams[3*2+0];
    Vthi=NeuronParams[3*2+1];
    Vree=NeuronParams[4*2+0];
    Vrei=NeuronParams[4*2+1];
    Vlbe=NeuronParams[5*2+0];
    Vlbi=NeuronParams[5*2+1];
    taurefe=NeuronParams[6*2+0];
    taurefi=NeuronParams[6*2+1];
    Iappe=NeuronParams[7*2+0];
    Iappi=NeuronParams[7*2+1];
    DeltaTe=NeuronParams[8*2+0];
    DeltaTi=NeuronParams[8*2+1];
    VTe=NeuronParams[9*2+0];
    VTi=NeuronParams[9*2+1];
    ae=NeuronParams[10*2+1];
    ai=NeuronParams[10*2+1];
    be=NeuronParams[11*2+1];
    bi=NeuronParams[11*2+1];
    tauwe=NeuronParams[12*2+1];
    tauwi=NeuronParams[12*2+1];
    
    tauf=mxGetScalar(prhs[14]);
    taue=mxGetScalar(prhs[15]);
    taui=mxGetScalar(prhs[16]);
    
    v0=mxGetPr(prhs[17]);
    m1 = mxGetM(prhs[17]);
    m2 = mxGetN(prhs[17]); 
    if(m1*m2!=N && m1*m2!=0)
        mexErrMsgTxt("V0 should be Nx1, 1xN or empty.");
    if(m1*m2==0){
        for(j=0;j<N;j++)
            if(NeuronType[j]==0)
                v0[j]=Vree;
            else
                v0[j]=Vrei;
    }
    
    dt=mxGetScalar(prhs[18]);
    Nt=(int)round(T/dt);
    
    /* Is there a stimulus. */
    if(m1Istim*m2Istim==0)
        IsStim=0;
    else{
        IsStim=1;        
        if(m1Istim*m2Istim!=Nt)
            mexErrMsgTxt("Istim should be empty, Ntx1 or 1xNt.");
    }

    
    maxns=(int)round(mxGetScalar(prhs[19]));
 

    Tburn=mxGetScalar(prhs[20]);
    if(Tburn<0 || Tburn>=T)
        mexErrMsgTxt("Tburn should be between 0 and T");
        
    sigma=mxGetPr(prhs[21]);
    m1 = mxGetM(prhs[21]);
    m2 = mxGetN(prhs[21]); 
    if(m1*m2==0)
        IsNoise=0;
    else
        if(m1*m2!=N)
           mexErrMsgTxt("Noise should be Nx1, 1xN or empty.");

    

    temp=mxGetPr(prhs[22]);
    m1 = mxGetM(prhs[22]);
    m2 = mxGetN(prhs[22]);     
    Nrecord=m1*m2;
    Irecord=mxMalloc(Nrecord*sizeof(int));
    for(j=0;j<Nrecord;j++){
        Irecord[j]=(int)round(temp[j])-1;
        if(Irecord[j]<0 || Irecord[j]>N-1)
            mexErrMsgTxt("Bad index in Irecord.");
        if(j>0)
            if(Irecord[j]<=Irecord[j-1]){
                mexPrintf("\n%d %d %d\n",j,Irecord[j],Irecord[j-1]);
                mexErrMsgTxt("Indices in Irecord should be in increasing order");
            }
    }  
    
    
        
    
    navgrecord=(int)round(mxGetScalar(prhs[23]));
    if(navgrecord<=0 && Nrecord>0){
        mexPrintf("%d",navgrecord);
        mexErrMsgTxt("navgrecord should be a positive int");
        
    }
    
        
    nburn=round(Tburn/dt);
    
    if(navgrecord<=0 && Nrecord>0)
        mexErrMsgTxt("navgrecord must be a pos int when Nrecord>0");
            
    if(Nrecord==0)
        Ntrecord=0;
    else
        Ntrecord=Nt/navgrecord;
    

    /* Allocate output vectors */
    plhs[0] = mxCreateDoubleMatrix(2, maxns, mxREAL);
    s=mxGetPr(plhs[0]);
    
    plhs[1] = mxCreateDoubleMatrix(Nrecord, Ntrecord, mxREAL);
    alphafr=mxGetPr(plhs[1]);
    
    plhs[2] = mxCreateDoubleMatrix(Nrecord, Ntrecord, mxREAL);
    alphaer=mxGetPr(plhs[2]);
    
    plhs[3] = mxCreateDoubleMatrix(Nrecord, Ntrecord, mxREAL);
    alphair=mxGetPr(plhs[3]);
    
    plhs[4] = mxCreateDoubleMatrix(Nrecord, Ntrecord, mxREAL);
    vr=mxGetPr(plhs[4]);
    
    plhs[5] = mxCreateDoubleMatrix(Nrecord, Ntrecord, mxREAL);
    wr=mxGetPr(plhs[5]);
     
    plhs[6] = mxCreateDoubleMatrix(Nrecord, Ntrecord, mxREAL);
    noiser=mxGetPr(plhs[6]);   
    
    plhs[7] = mxCreateDoubleMatrix(N, 1, mxREAL);
    malphafr=mxGetPr(plhs[7]);
    
    plhs[8] = mxCreateDoubleMatrix(N, 1, mxREAL);
    malphaer=mxGetPr(plhs[8]);
    
    plhs[9] = mxCreateDoubleMatrix(N, 1, mxREAL);
    malphair=mxGetPr(plhs[9]);
    
    plhs[10] = mxCreateDoubleMatrix(N, 1, mxREAL);
    mvr=mxGetPr(plhs[10]);

    plhs[11] = mxCreateDoubleMatrix(N, 1, mxREAL);
    mwr=mxGetPr(plhs[11]);    
  
   
    refstate=mxMalloc(N*sizeof(int));
    JnextE=mxMalloc(N*sizeof(double));
    JnextI=mxMalloc(N*sizeof(double));
    JnextF=mxMalloc(N*sizeof(double));
    alphae=mxMalloc(N*sizeof(double));
    alphai=mxMalloc(N*sizeof(double));
    alphaf=mxMalloc(N*sizeof(double));
    w=mxMalloc(N*sizeof(double));
    v=mxMalloc(N*sizeof(double));
    
    /* Inititalize variables */
    for(j=0;j<N;j++){
        v[j]=v0[j]; 
        refstate[j]=0;
        JnextE[j]=0;
        JnextI[j]=0;
        JnextF[j]=0;
        alphae[j]=0;
        alphai[j]=0;
        alphaf[j]=0;
        w[j]=0;
    }
    

    /* Refractory states */
    Ntrefe=(int)round(taurefe/dt);
    Ntrefi=(int)round(taurefi/dt);
    
    printfperiod=(int)(round(Nt/10.0));

    /* Initialize number of spikes */
    ns=0;

    mexEvalString("drawnow;");
    
    iFspike=0;
    noise=0;noise1=0;tempstim=0;tempnoise=0;
    /* Main loop */
    /* Exit loop and issue a warning if max number of spikes is exceeded */
    for(i=0;i<Nt && ns<maxns;i++){
        
        jrecord=0;
    
        /* Propagate ffwd spikes */
        if(isF){
            while(sf[iFspike*2+0]<=i*dt && iFspike<Nsf){   /* Find spikes in this time bin */
                jf=(int)round(sf[iFspike*2+1]); /* Presynaptic cell index (in ffwd layer)*/
                for(k=Jf0[jf];k<Jf0[jf+1];k++)  
                    JnextF[(int)round(Jfinds[k])-1]+=Jfweights[k];  /* Increment postsynaptic Jnext */
                iFspike++; /* number of ffwd spikes accounted for */
            }
        }
        
      /* Iterate through recurrent network */
     for(j=0;j<N;j++){    
             

                    
        if(IsStim)
            tempstim=Istim[i]*JIstim[j];

         
        /* Make Gaussian rv using Box-Muller (sp?) */
        /* This makes two random nums, so store the */
        /* previous one for odd iterations */
        if(IsNoise){
        if(j%2==0){
            u1=drand48();
            u2=drand48();

            noise=sqrt(-2*log(u1))*cos(2*M_PI*u2);
            noise1=sqrt(-2*log(u1))*sin(2*M_PI*u2);

        }
        else
            noise=noise1;
        
        tempnoise=sigma[j]*noise/sqrt(dt);
        }
        
        
     
         
         /* If cell is exc */
         if(NeuronType[j]==0){
             
             w[j]+=(dt/tauwe)*(-w[j]+ae*(v[j]-ELe));
             if(refstate[j]<=0)
                 
                
                /* Euler step for mem pot */                    
                v[j]+=fmax((tempstim+tempnoise+alphae[j]+alphai[j]+alphaf[j]-gLe*(v[j]-ELe)+gLe*DeltaTe*exp((v[j]-VTe)/DeltaTe)-w[j])*dt/Ce,Vlbe-v[j]);
             else{                 
                 v[j]=Vree;
                 refstate[j]--;
             }

              /* If a spike occurs */
              if(v[j]>=Vthe && refstate[j]==0 && ns<maxns){

                  refstate[j]=Ntrefe;
                  w[j]+=be;
                  v[j]=Vree;       /* reset membrane potential */
                  s[0+2*ns]=i*dt; /* spike time */
                  s[1+2*ns]=j+1;     /* neuron index */
                  ns++;           /* update total number of spikes */

                  /* For each postsynaptic target, propagate spike into JnextE */
                  if(IsRec){
                      for(k=J0[j];k<J0[j+1];k++)  
                          JnextE[(int)round(Jinds[k])-1]+=Jweights[k];  /* Increment postsynaptic Jnext */
                  }

              }
             
         }         
         else{ /* If it's an inhibitory neuron */

             w[j]+=(dt/tauwi)*(-w[j]+ai*(v[j]-ELi));
             if(refstate[j]<=0)
                /* Euler step for mem pot */
                v[j]+=fmax((tempstim+tempnoise+alphae[j]+alphai[j]+alphaf[j]-gLi*(v[j]-ELi)+gLi*DeltaTi*exp((v[j]-VTi)/DeltaTi)-w[j])*dt/Ci,Vlbi-v[j]);
             else{                 
                 v[j]=Vrei;
                 refstate[j]--;
             }

              /* If a spike occurs */
              if(v[j]>=Vthi && refstate[j]==0 && ns<maxns){

                  refstate[j]=Ntrefi;
                  w[j]+=bi;
                  v[j]=Vrei;       /* reset membrane potential */
                  s[0+2*ns]=i*dt; /* spike time */
                  s[1+2*ns]=j+1;     /* neuron index */
                  ns++;           /* update total number of spikes */

                  /* For each postsynaptic target, propagate spike into JnextE */
                  if(IsRec){
                      for(k=J0[j];k<J0[j+1];k++)  
                          JnextI[(int)round(Jinds[k])-1]+=Jweights[k];  /* Increment postsynaptic Jnext */
                  }

              }
             
         }
         
         /* Compute cumulative sums for means */
         if(i>nburn){
             malphaer[j]+=alphae[j];
             malphair[j]+=alphai[j];
             malphafr[j]+=alphaf[j]+tempstim;
             mvr[j]+=v[j];
             mwr[j]+=w[j];
         }
         
      
        /* Store recorded variables */
         if(jrecord<Nrecord){
         if(j==Irecord[jrecord] && i/navgrecord<Ntrecord){
           ii=i/navgrecord;           
           alphaer[jrecord+Nrecord*ii]+=alphae[j]/navgrecord;
           alphair[jrecord+Nrecord*ii]+=alphai[j]/navgrecord;
           alphafr[jrecord+Nrecord*ii]+=(alphaf[j]+tempstim)/navgrecord;
           vr[jrecord+Nrecord*ii]+=v[j]/navgrecord;
           wr[jrecord+Nrecord*ii]+=w[j]/navgrecord;
           noiser[jrecord+Nrecord*ii]+=tempnoise/navgrecord;
           jrecord++;
          }
         }
         

         /* Euler step for currents */
         alphae[j]-=alphae[j]*(dt/taue);
         alphai[j]-=alphai[j]*(dt/taui);
         alphaf[j]-=alphaf[j]*(dt/tauf);
         
        }  /* End main j loop over N */  
         
      
        
        
        /* Use Jnext vectors to update synaptic variables */
        for(j=0;j<N;j++){                     
          alphae[j]+=JnextE[j]/taue;
          alphai[j]+=JnextI[j]/taui;
          alphaf[j]+=JnextF[j]/tauf;
          JnextE[j]=0;
          JnextI[j]=0; 
          JnextF[j]=0;
          
        } 
        

        
        
         /*
          if(i%printfperiod==0){
             mexPrintf("\n%d percent complete  rate = %2.2fHz",i*100/Nt,1000*((double)(ns))/(((double)(N))*((double)(i))*dt));
             mexEvalString("drawnow;");
         }
         */
                
    } /* End i loop over time */

/* compute means */
for(jj=0;jj<N;jj++){
   malphaer[jj]=malphaer[jj]/(Nt-nburn);
   malphair[jj]=malphair[jj]/(Nt-nburn);
   malphafr[jj]=(malphafr[jj])/(Nt-nburn);
   mvr[jj]=mvr[jj]/(Nt-nburn);
   mwr[jj]=mwr[jj]/(Nt-nburn);
  } 
    
    

    for(j=0;j<Nsf;j++)
        sf[j*2+1]=sf[j*2+1]+1;
            

    /* Issue a warning if max number of spikes reached */
    if(ns>=maxns)
       mexWarnMsgTxt("Maximum number of spikes reached, simulation terminated.");

/*
    for(j=0;j<Nsf;j++)
        sf[j*2+1]=sf[j*2+1]+1;
*/  

    mxFree(NeuronType);   
    mxFree(J0);   
    mxFree(Jf0); 
    mxFree(Irecord);          
    mxFree(refstate);
    mxFree(JnextE);
    mxFree(JnextI);
    mxFree(JnextF);    
    mxFree(alphae);
    mxFree(alphai);
    mxFree(alphaf);
    mxFree(v);   
    mxFree(w);  
    
    

    
}

