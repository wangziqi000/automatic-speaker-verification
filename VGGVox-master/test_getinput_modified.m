function inp = test_getinput_modified(snd,meta,buckets)
    
	SPEC 		= runSpec(snd,meta.audio);
	mu    		= mean(SPEC,2);
    stdev 		= std(SPEC,[],2) ;
    nSPEC 		= bsxfun(@minus, SPEC, mu);
    nSPEC 		= bsxfun(@rdivide, nSPEC, stdev);

    rsize 	= buckets.width(find(buckets.width(:)<=size(nSPEC,2),1,'last'));
    rstart  = round((size(nSPEC,2)-rsize)/2);
    if rstart == 0
        rstart = 1;
    end

% 	inp(:,:) = gpuArray(single(nSPEC(:,rstart:rstart+rsize-1)));
    inp(:,:) = single(nSPEC(:,rstart:rstart+rsize-1));

end 

