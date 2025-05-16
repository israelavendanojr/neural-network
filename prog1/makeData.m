%% makeData(seed, featDim, polyOrder, stdev, nTrain, nDev, outdir)
%     - seed:      the seed for the random number generator
%     - featDim:   the dimension, D, of the inputs to create
%     - polyOrder: the order of the polynomial of the underyling true relationship
%                  note that for this script polyOrder can only be >1 if featDim=1
%     - stdev:     the standard deviation of the noise that corrupts outputs y
%     - nTrain:    the number of training set samples to generate
%     - nDev:      the number of dev set samples to generate
%     - outdir:    the name of an *existing* directory where the results will be written
%        
% Upon completion, the following files will be created
%    outdir/{train,dev}_{x,y}.txt
%
% Brian Hutchinson
%
function [] = makeData(seed,featDim,polyOrder,stdev,nTrain,nDev,outdir)
	if( polyOrder > 1 && featDim > 1 )
		fprintf('Error: true function can be polynomial or vector valued but not both.\n');
		exit();
	end

	rand("seed",seed); % set the random number generator seed

	D = max(featDim,polyOrder);

	true_beta = randn(D+1,1) % generate the parameters for the true underlying relationship

	f = @(x) ([1,x]*true_beta);

	[tr_x,tr_y] = generateDataset(nTrain,featDim,polyOrder,f,stdev);
	[de_x,de_y] = generateDataset(nDev,featDim,polyOrder,f,stdev);

	saveData(tr_x,sprintf('%s/train_x.txt',outdir));
	saveData(tr_y,sprintf('%s/train_y.txt',outdir));
	saveData(de_x,sprintf('%s/dev_x.txt',outdir));
	saveData(de_y,sprintf('%s/dev_y.txt',outdir));
end

% helper routine to generate random input and output pairs
function [x,y] = generateDataset(nSamples,D,K,f,stdev)
	x = zeros(nSamples,D);
	y = zeros(nSamples,1);
	for i=1:nSamples
		x(i,:) = randn(1,D);
		for j=1:K
			x(i,j) = x(i,1)^j;	
		end
		y(i) = f(x(i,:)) + randn(1)*stdev;
	end
	x = x(:,1:D);
end

% helper routine to save matrices to file
function [] = saveData(X,filename)
	fid = fopen(filename,'w');
	for i=1:size(X,1)
		for j=1:size(X,2)
			fprintf(fid,'%.5f ',X(i,j));
		end
		fprintf(fid,'\n');
	end

	fclose(fid);
end
