addpath('..\Data\awdEvents1');
addpath('..\Data\marion');
addpath('..\Data\sanae');
nframe=2000;
stereo=2;
fh=fopen('2012-02-01UT02_37_22.52623133.sanae.vr2');
[wh,nframe_read,orig_fr]=frread(fh,nframe,stereo);