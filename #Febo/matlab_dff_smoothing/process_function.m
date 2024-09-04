%Implementation of Nature protocol
%Hongbo Jia, Nathalie L Rochefort1, Xiaowei Chen & Arthur Konnerth1 "In
%vivo two-photon imaging of sensory-evoked dendritic calcium signals in cortical neurons"
%
%Implementation copyright Petros Xanthopoulos 2013-2014
%usage: signalout=process_function(signalin,t_0,t_1,t_2)
% where
% input: signalin is the raw signal 
%t_0,t_1,t_2 are the parameters described in Nature protocol paper
%comments: for a 30Hz imaging systems the following parameter setup is
%recommended (empirical note on Nature paper): 
%t_0= 0.2;
%t_1=0.75;
%t_2=3;

% 9/24/16 - Added signalout_dff (R_0) output (i.e. deltaF/F prior to EWMA filtering); JEC Corrected an error (replaced F_sm with signalin)


function [signalout_dff,signalout_ewma]=process_function_jc(signalin,t_0,t_1,t_2,samplingfreq)

F_0=[];

Fs=samplingfreq; %sampling frequency

t_0_s=floor(t_0*Fs);
t_1_s=floor(t_1*Fs);
t_2_s=floor(t_2*Fs);

F_sm = smooth(signalin,t_1_s);

for i=(t_2_s+1):length(signalin)
    F_0=[F_0 min(F_sm(i-t_2_s:i))];
end

start = 1+length(signalin)-length(F_0);

R_0 = (signalin((t_2_s+1):end)-F_0)./F_0;

R_0_sm = EWMA(R_0,t_0_s);

signalout_dff = R_0;
signalout_ewma = R_0_sm;


