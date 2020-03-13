function[features] = vcc_vox_net_x1_vgg(snd)
%% Set up VGGVox
    addpath("VGGVox-master")
    setup_VGGVox()
   
    opts.modelPath = '' ;
    % opts.gpu = 3;
    opts.gpu = 0;
    
    % Load or download the VGGVox model for Verification
    modelName = 'vggvox_ver_net.mat' ;
    paths = {opts.modelPath, ...
        modelName, ...
        fullfile(vl_rootnn, 'data', 'models-import', modelName)} ;
    ok = find(cellfun(@(x) exist(x, 'file'), paths), 1) ;

    if isempty(ok)
        fprintf('Downloading the VGGVox model for Verification ... this may take a while\n') ;
        opts.modelPath = fullfile(vl_rootnn, 'data/models-import', modelName) ;
        mkdir(fileparts(opts.modelPath)) ; base = 'http://www.robots.ox.ac.uk' ;
        url = sprintf('%s/~vgg/data/voxceleb/models/%s', base, modelName) ;
        urlwrite(url, opts.modelPath) ;
    else
        opts.modelPath = paths{ok} ;
    end
    load(opts.modelPath); net = dagnn.DagNN.loadobj(netStruct);

    % Remove loss layers and add distance layer
    names = {'loss'} ;
    for i = 1:numel(names)
        layer = net.layers(net.getLayerIndex(names{i})) ;
        net.removeLayer(names{i}) ;
        net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
    end
    net.addLayer('dist', dagnn.PDist('p',2), {'x1_s1', 'x1_s2'}, 'distance');


   
%% Set up audio

    snd = resample(snd, 320, 441); % from 22050 to 16000 

    opt.audio.window   = [0 1];
    opt.audio.fs       = 16000;
    opt.audio.Tw       = 25;
    opt.audio.Ts       = 10;            % analysis frame shift (ms)
    opt.audio.alpha    = 0.97;          % preemphasis coefficient
    opt.audio.R        = [];  % frequency range to consider
    opt.audio.M        = 40;            % number of filterbank channels
    opt.audio.C        = [];            % number of cepstral coefficients
    opt.audio.L        = [];            % cepstral sine lifter parameter%keyboard;


    net.meta = opt; 

    % Evaluate network on CPU and set it to test mode
    net.conserveMemory = 0;
    net.mode = 'test' ;

    % Setup buckets to allow for average pooling 
    buckets.pool 	= [2 5 8 11 14 17 20 23 27 30];
    buckets.width 	= [100 200 300 400 500 600 700 800 900 1000];

    % Load input pair and do a forward pass
    inp1 = test_getinput_modified(snd, net.meta, buckets);

    s1 = size(inp1,2);

    p1 = buckets.pool(s1==buckets.width);

    net.layers(22).block.poolSize=[1 p1];

    net.eval({ 'input_b1', inp1 });

    featid = strcmp({net.vars.name},'x1_s1');
    features = squeeze(net.vars(featid).value);