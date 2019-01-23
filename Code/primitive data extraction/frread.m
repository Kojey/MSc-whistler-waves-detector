function [wh,nframe_read,orig_fr]=frread(fh,nframe,stereo)
% function [wh,nframe_read,orig_fr]=frread(fname,nframe,stereo)
%function [wh,nframe_read,orig_fr]=frread_fftvr2(fh,nframe,stereo)
%
% megnyitott (fh) vr2 eredeti file-bol beolvas nframe frame-et
% az nframe frame-et az orig_fr tombbe teszi
% frame hossz

% INPUTS:
% 	fh		file handle
%	nframe 	Number of frames to read. Set to arb 'large' number, as fread will stop at EOF.
% 	stereo	Number of components (SAN, MAR: stereo=1; GRA: stereo=0)

% 
% fh = fopen(fname,'rb');

% ???
frlen=4103;

%megnezzuk, van-e ennyi frame a file-ban. 

%% ftell(fh) returns the current location of the position pointer in the specified file (fh)
itt=ftell(fh);

%% fseek(fh,p0,p1)
%% sets the file position indicator p0 bytes from p1 in the specified file (fh)
fseek(fh,0,1); % seek to EOF (=1) from origin 0

% get current position
true_len=ftell(fh);

% go beginning of file (-1 position is BOF)
fseek(fh,itt,-1);

% if true_len is smaller than 2 x frame_length x number of frames
%% if not at end of file, then make the number of frames (file len)/ (2*framelen) 
if(true_len<2*frlen*nframe)
  disp(true_len)
	nframe=floor(true_len/2/frlen)
end

disp('nframe')
disp(nframe)

if stereo
	adatlen=2048;
	wh=zeros(2,floor(nframe*adatlen)); 
%	[orig_fr,nword_read]=fread(fh,nframe*frlen,'char');
%	save qq orig_fr;

	%% read file fh, of length (num of frames X framelength), 
  %% with precision 'short' (signed, 16 bit, i.e. +- 1e15)
 	[orig_fr,nword_read]=fread(fh,nframe*frlen,'short');
 	%% transpose 
	orig_frt=orig_fr';
  
  disp(nword_read)
  origFrLen = length(orig_fr)

	% number of frames to read
	nframe_read=nword_read/frlen;

	% for every frame
	for i=1:nframe_read
		% elements of comp 1
		i1=8+(i-1)*frlen:2:i*frlen;	
		% elements of comp 2
		i2=9+(i-1)*frlen:2:i*frlen;
		% columns to use
		ii=(i-1)*adatlen+1:i*adatlen;
		% comp 1
		wh(1,ii)=orig_frt(i1);
		% comp 2
		wh(2,ii)=orig_frt(i2);
	end
else
	adatlen=4096;
	wh=zeros(1,floor(nframe*adatlen)); 
	[orig_fr,nword_read]=fread(fh,nframe*frlen,'short');
	orig_frt=orig_fr';

	nframe_read=nword_read/frlen;

	for i=1:nframe_read
  	i1=8+(i-1)*frlen:i*frlen;	
  	ii=(i-1)*adatlen+1:i*adatlen;	
	wh(1,ii)=orig_frt(i1);
	end
end

fclose(fh)
